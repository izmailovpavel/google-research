# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run an Hamiltonian Monte Carlo chain on a cloud TPU."""

import os
import numpy as onp
from jax import numpy as jnp
import jax
import tensorflow.compat.v2 as tf
import argparse
import time

from bnn_hmc.core import data
from bnn_hmc.core import losses
from bnn_hmc.core import models
from bnn_hmc.utils import checkpoint_utils
from bnn_hmc.utils import cmd_args_utils
from bnn_hmc.utils import logging_utils
from bnn_hmc.utils import train_utils
from bnn_hmc.utils import tree_utils
from bnn_hmc.utils import metrics
from bnn_hmc.utils import ensemble_utils

parser = argparse.ArgumentParser(description="Run an HMC chain on a cloud TPU")
cmd_args_utils.add_common_flags(parser)
parser.add_argument("--step_size", type=float, default=1.e-4,
                    help="HMC step size")
parser.add_argument("--burn_in_step_size_factor", type=float, default=1.,
                    help="Multiplicative factor by which step size is re-scaled"
                         "during burn-in phase")
parser.add_argument("--step_size_adaptation_speed", type=float, default=0.,
                    help="Step size adaptation speed")
parser.add_argument("--target_accept_rate", type=float, default=0.8,
                    help="Target accept rate in the M-H correction step")
parser.add_argument("--trajectory_len", type=float, default=1.e-3,
                    help="HMC trajectory length")
parser.add_argument("--num_iterations", type=int, default=1000,
                    help="Total number of HMC iterations")
parser.add_argument("--max_num_leapfrog_steps", type=int, default=10000,
                    help="Maximum number of leapfrog steps allowed; increase to"
                         "run longer trajectories")
parser.add_argument("--num_burn_in_iterations", type=int, default=0,
                    help="Number of burn-in iterations")
parser.add_argument("--no_mh", default=False, action='store_true',
                    help="If set, Metropolis Hastings correction is ignored")

args = parser.parse_args()
train_utils.set_up_jax(args.tpu_ip, args.use_float64)
  
  
def train_model():
  
  subdirname = (
    "model_{}_wd_{}_stepsize_{}_trajlen_{}_burnin_{}_{}_mh_{}_temp_{}_"
    "seed_{}".format(
      args.model_name, args.weight_decay, args.step_size, args.trajectory_len,
      args.num_burn_in_iterations, args.burn_in_step_size_factor,
      not args.no_mh, args.temperature, args.seed
    ))
  dirname = os.path.join(args.dir, subdirname)
  os.makedirs(dirname, exist_ok=True)
  tf_writer = tf.summary.create_file_writer(dirname)
  cmd_args_utils.save_cmd(dirname, tf_writer)
  num_devices = len(jax.devices())

  dtype = jnp.float64 if args.use_float64 else jnp.float32
  train_set, test_set, num_classes = data.make_ds_pmap_fullbatch(
    args.dataset_name, dtype)

  net_apply, net_init = models.get_model(args.model_name, num_classes)
  
  checkpoint_dict, status = checkpoint_utils.initialize(
      dirname, args.init_checkpoint)
  
  if status == checkpoint_utils.InitStatus.LOADED_PREEMPTED:
    print("Continuing the run from the last saved checkpoint")
    (start_iteration, params, net_state, key, step_size, _, num_ensembled,
     ensemble_predicted_probs) = (
        checkpoint_utils.parse_hmc_checkpoint_dict(checkpoint_dict))
    
  else:
    key, net_init_key = jax.random.split(jax.random.PRNGKey(args.seed), 2)
    start_iteration = 0
    num_ensembled = 0
    ensemble_predicted_probs = None
    step_size = args.step_size
    
    if status == checkpoint_utils.InitStatus.INIT_CKPT:
      print("Resuming the run from the provided init_checkpoint")
      _, params, net_state, _, _, _, _, _ = (
        checkpoint_utils.parse_hmc_checkpoint_dict(checkpoint_dict))
    elif status == checkpoint_utils.InitStatus.INIT_RANDOM:
      print("Starting from random initialization with provided seed")
      key, net_init_key = jax.random.split(jax.random.PRNGKey(args.seed), 2)
      init_data = jax.tree_map(lambda elem: elem[0][:1], train_set)
      params, net_state = net_init(net_init_key, init_data, True)
      net_state = jax.pmap(lambda _: net_state)(jnp.arange(num_devices))
    else:
      raise ValueError("Unknown initialization status: {}".format(status))

  # manually convert all params to dtype
  params = jax.tree_map(lambda p: p.astype(dtype), params)
  
  param_types = tree_utils.tree_get_types(params)
  assert all([p_type == dtype for p_type in param_types]), (
    "Params data types {} do not match specified data type {}".format(
      param_types, dtype))
    
  trajectory_len = args.trajectory_len

  log_likelihood_fn = losses.make_xent_log_likelihood(
    num_classes, args.temperature)
  log_prior_fn, log_prior_diff_fn = losses.make_gaussian_log_prior(
    args.weight_decay, args.temperature)

  update, get_log_prob_and_grad, evaluate = train_utils.make_hmc_update(
    net_apply, log_likelihood_fn, log_prior_fn, log_prior_diff_fn,
    args.max_num_leapfrog_steps, args.target_accept_rate,
    args.step_size_adaptation_speed)

  log_prob, state_grad, log_likelihood, net_state = (
      get_log_prob_and_grad(train_set, params, net_state))

  assert log_prob.dtype == dtype, (
    "log_prob data type {} does not match specified data type {}".format(
        log_prob.dtype, dtype))

  grad_types = tree_utils.tree_get_types(state_grad)
  assert all([g_type == dtype for g_type in grad_types]), (
    "Gradient data types {} do not match specified data type {}".format(
      grad_types, dtype))

  ensemble_acc = 0
  
  for iteration in range(start_iteration, args.num_iterations):
    
    # do a linear ramp-down of the step-size in the burn-in phase
    if iteration < args.num_burn_in_iterations:
      alpha = iteration / (args.num_burn_in_iterations - 1)
      initial_step_size = args.step_size
      final_step_size = args.burn_in_step_size_factor * args.step_size
      step_size = final_step_size * alpha + initial_step_size * (1 - alpha)
    in_burnin = (iteration < args.num_burn_in_iterations)
    do_mh_correction = (not args.no_mh) and (not in_burnin)

    start_time = time.time()
    (params, net_state, log_likelihood, state_grad, step_size, key,
     accept_prob, accepted) = (
        update(train_set, params, net_state, log_likelihood, state_grad,
               key, step_size, trajectory_len, do_mh_correction))
    iteration_time = time.time() - start_time

    checkpoint_name = checkpoint_utils.make_checkpoint_name(iteration)
    checkpoint_path = os.path.join(dirname, checkpoint_name)
    checkpoint_dict = checkpoint_utils.make_hmc_checkpoint_dict(
        iteration, params, net_state, key, step_size, accepted, num_ensembled,
        ensemble_predicted_probs)
    checkpoint_utils.save_checkpoint(checkpoint_path, checkpoint_dict)
      
    test_stats = evaluate(params, net_state, test_set)
    train_stats = evaluate(params, net_state, train_set)
    del test_stats["prior"]
    tabulate_dict = logging_utils.make_common_logs(
        tf_writer, iteration, iteration_time, train_stats, test_stats)

    if ((not in_burnin) and accepted) or args.no_mh:
      ensemble_predicted_probs, num_ensembled, ensemble_stats = (
          ensemble_utils.update_ensemble_classification(
              net_apply, params, net_state, test_set, num_ensembled,
              ensemble_predicted_probs))
      logging_utils.add_ensemle_logs(
          tf_writer, tabulate_dict, ensemble_stats, iteration)

    tabulate_dict["accept_prob"] = accept_prob
    tabulate_dict["accepted"] = accepted

    with tf_writer.as_default():
      tf.summary.scalar("telemetry/accept_prob", accept_prob, step=iteration)
      tf.summary.scalar("telemetry/accepted", accepted, step=iteration)
      
      tf.summary.scalar("hypers/step_size", step_size, step=iteration)
      tf.summary.scalar("hypers/trajectory_len", trajectory_len,
                        step=iteration)
      tf.summary.scalar("hypers/weight_decay", args.weight_decay,
                        step=iteration)
      tf.summary.scalar("hypers/temperature", args.temperature,
                        step=iteration)

      tf.summary.scalar("debug/do_mh_correction", float(do_mh_correction),
                        step=iteration)
      tf.summary.scalar("debug/in_burnin", float(in_burnin),
                        step=iteration)

    table = logging_utils.make_table(
      tabulate_dict, iteration - start_iteration, args.tabulate_freq)
    print(table)


if __name__ == "__main__":
  print("JAX sees the following devices:", jax.devices())
  train_model()

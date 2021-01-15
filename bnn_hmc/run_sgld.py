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

"""Run an SGLD chain on a cloud TPU. We are not using data augmentation."""

import os
from jax import numpy as jnp
import jax
import tensorflow.compat.v2 as tf
import argparse
import time
import numpy as onp
from collections import OrderedDict

from core import sgmcmc
from bnn_hmc.utils import checkpoint_utils
from bnn_hmc.utils import cmd_args_utils
from bnn_hmc.utils import logging_utils
from bnn_hmc.utils import precision_utils
from bnn_hmc.utils import train_utils
from bnn_hmc.utils import tree_utils
from bnn_hmc.utils import metrics
from bnn_hmc.utils import data_utils
from bnn_hmc.utils import models
from bnn_hmc.utils import losses

parser = argparse.ArgumentParser(description="Run SGD on a cloud TPU")
cmd_args_utils.add_common_flags(parser)
parser.add_argument("--init_step_size", type=float, default=1.e-7,
                    help="Initial SGD step size")
parser.add_argument("--final_step_size", type=float, default=5.e-8,
                    help="Initial SGD step size")
parser.add_argument("--num_epochs", type=int, default=1000,
                    help="Total number of SGD epochs iterations")
parser.add_argument("--num_burnin_epochs", type=int, default=300,
                    help="Number of epochs before final lr is reached")
parser.add_argument("--batch_size", type=int, default=80, help="Batch size")
parser.add_argument("--eval_freq", type=int, default=10,
                    help="Frequency of evaluation (epochs)")
parser.add_argument("--save_freq", type=int, default=50,
                    help="Frequency of checkpointing (epochs)")
parser.add_argument("--ensemble_freq", type=int, default=10,
                    help="Frequency of checkpointing (epochs)")


args = parser.parse_args()
train_utils.set_up_jax(args.tpu_ip, args.use_float64)


def train_model():
  subdirname = (
    "sgld_wd_{}_stepsizes_{}_{}_batchsize_{}_epochs{}_{}_temp_{}_seed_{}".format(
    args.weight_decay, args.init_step_size, args.final_step_size,
    args.batch_size, args.num_epochs, args.num_burnin_epochs,
    args.temperature, args.seed))
  dirname = os.path.join(args.dir, subdirname)
  os.makedirs(dirname, exist_ok=True)
  tf_writer = tf.summary.create_file_writer(dirname)
  cmd_args_utils.save_cmd(dirname, tf_writer)

  dtype = jnp.float64 if args.use_float64 else jnp.float32
  train_set, test_set, task, data_info = data_utils.make_ds_pmap_fullbatch(
    args.dataset_name, dtype)

  net_apply, net_init = models.get_model(args.model_name, data_info)
  net_apply = precision_utils.rewrite_high_precision(net_apply)

  (likelihood_factory, predict_fn, ensemble_upd_fn, metrics_fns,
   tabulate_metrics) = train_utils.get_task_specific_fns(task, data_info)
  log_likelihood_fn = likelihood_factory(args.temperature)
  log_prior_fn, log_prior_diff_fn = losses.make_gaussian_log_prior(
    args.weight_decay, args.temperature)

  num_data = jnp.size(train_set[1])
  num_batches = num_data // args.batch_size
  num_devices = len(jax.devices())
  
  burnin_steps = num_batches * args.num_burnin_epochs
  lr_schedule = train_utils.make_cosine_lr_schedule_with_burnin(
      args.init_step_size, args.final_step_size, burnin_steps
  )
  optimizer = sgmcmc.sgld_gradient_update(lr_schedule, args.seed)
  
  checkpoint_dict, status = checkpoint_utils.initialize(
    dirname, args.init_checkpoint)

  if status == checkpoint_utils.InitStatus.LOADED_PREEMPTED:
    print("Continuing the run from the last saved checkpoint")
    (start_iteration, params, net_state, opt_state, key, num_ensembled,
     _, ensemble_predicted_probs) = (
      checkpoint_utils.parse_sgmcmc_checkpoint_dict(checkpoint_dict))
    
  else:
    key, net_init_key = jax.random.split(jax.random.PRNGKey(args.seed), 2)
    start_iteration = 0
    num_ensembled = 0
    ensemble_predictions = None
    key = jax.random.split(key, num_devices)
  
    if status == checkpoint_utils.InitStatus.INIT_CKPT:
      print("Resuming the run from the provided init_checkpoint")
      _, params, net_state, _, _, _, _ = (
        checkpoint_utils.parse_sgmcmc_checkpoint_dict(checkpoint_dict))
      opt_state = optimizer.init(params)
    elif status == checkpoint_utils.InitStatus.INIT_RANDOM:
      print("Starting from random initialization with provided seed")
      init_data = jax.tree_map(lambda elem: elem[0][:1], train_set)
      params, net_state = net_init(net_init_key, init_data, True)
      opt_state = optimizer.init(params)
      net_state = jax.pmap(lambda _: net_state)(jnp.arange(num_devices))
    else:
      raise ValueError("Unknown initialization status: {}".format(status))
    
  sgmcmc_train_epoch = train_utils.make_sgd_train_epoch(
    net_apply, log_likelihood_fn, log_prior_fn, optimizer, num_batches)

  param_types = tree_utils.tree_get_types(params)
  assert all([p_type == dtype for p_type in param_types]), (
    "Params data types {} do not match specified data type {}".format(
      param_types, dtype))
  
  for iteration in range(start_iteration, args.num_epochs):
    
    start_time = time.time()
    params, net_state, opt_state, logprob_avg, key = sgmcmc_train_epoch(
      params, net_state, opt_state, train_set, key)
    iteration_time = time.time() - start_time

    # Evaluation
    train_stats = {"log_prob": logprob_avg}
    test_stats = {}
    is_evaluation_epoch = (
      (iteration % args.eval_freq == 0) or (iteration == args.num_epochs - 1))
    is_ensembling_epoch = ((iteration > args.num_burnin_epochs) and
      ((iteration - args.num_burnin_epochs) % args.ensemble_freq == 0))

    if is_evaluation_epoch or is_ensembling_epoch:
      test_predictions = onp.asarray(
        predict_fn(net_apply, params, net_state, test_set))
      train_predictions = onp.asarray(
        predict_fn(net_apply, params, net_state, train_set))
      test_stats = train_utils.evaluate_metrics(
        test_predictions, test_set[1], metrics_fns)
      train_stats = train_utils.evaluate_metrics(
        train_predictions, train_set[1], metrics_fns)
      train_stats["prior"] = log_prior_fn(params)

    # Ensembling
    if is_ensembling_epoch:
      ensemble_predictions = ensemble_upd_fn(
        ensemble_predictions, num_ensembled, test_predictions)
      ensemble_stats = train_utils.evaluate_metrics(
        ensemble_predictions, test_set[1], metrics_fns)
      num_ensembled += 1
    else:
      ensemble_stats = {}
      test_predictions = None

    # Checkpoint
    if iteration % args.save_freq == 0 or iteration == args.num_epochs - 1:
      checkpoint_name = checkpoint_utils.make_checkpoint_name(iteration)
      checkpoint_path = os.path.join(dirname, checkpoint_name)
      checkpoint_dict = checkpoint_utils.make_sgmcmc_checkpoint_dict(
        iteration, params, net_state, opt_state, key, num_ensembled,
        test_predictions, ensemble_predictions)
      checkpoint_utils.save_checkpoint(checkpoint_path, checkpoint_dict)

    # Logging
    other_logs = {
      "telemetry/iteration": iteration,
      "telemetry/iteration_time": iteration_time,
      "telemetry/num_ensembled": num_ensembled,
      "hypers/step_size": lr_schedule(opt_state.count),
      "hypers/weight_decay": args.weight_decay,
      "hypers/temperature": args.temperature,
    }
    logging_dict = logging_utils.make_logging_dict(
      train_stats, test_stats, ensemble_stats)
    logging_dict.update(other_logs)

    with tf_writer.as_default():
      for stat_name, stat_val in logging_dict.items():
        tf.summary.scalar(stat_name, stat_val, step=iteration)
    tabulate_dict = OrderedDict()
    tabulate_dict["i"] = iteration
    tabulate_dict["t"] = iteration_time
    tabulate_dict["lr"] = lr_schedule(opt_state.count)
    for metric_name in tabulate_metrics:
      if metric_name in logging_dict:
        tabulate_dict[metric_name] = logging_dict[metric_name]
      else:
        tabulate_dict[metric_name] = None

    table = logging_utils.make_table(
      tabulate_dict, iteration - start_iteration, args.tabulate_freq)
    print(table)


if __name__ == "__main__":
  print("JAX sees the following devices:", jax.devices())
  print("TF sees the following devices:", tf.config.get_visible_devices())
  train_model()

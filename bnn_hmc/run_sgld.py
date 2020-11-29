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

from bnn_hmc import data
from bnn_hmc import models
from bnn_hmc import nn_loss
from bnn_hmc import train_utils
from bnn_hmc import checkpoint_utils
from bnn_hmc import cmd_args_utils
from bnn_hmc import tabulate_utils
from bnn_hmc import sgmcmc
from bnn_hmc import metrics
from bnn_hmc import precision_utils


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
  train_set, test_set, num_classes = data.make_ds_pmap_fullbatch(
    args.dataset_name, dtype)
  
  net_apply, net_init = models.get_model(args.model_name, num_classes)
  net_apply = precision_utils.rewrite_high_precision(net_apply)
  
  log_likelihood_fn = nn_loss.make_xent_log_likelihood(
      num_classes, args.temperature)
  log_prior_fn, _ = nn_loss.make_gaussian_log_prior(
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
     ensemble_predicted_probs) = (
      checkpoint_utils.parse_sgmcmc_checkpoint_dict(checkpoint_dict))
    
  else:
    key, net_init_key = jax.random.split(jax.random.PRNGKey(args.seed), 2)
    start_iteration = 0
    num_ensembled = 0
    ensemble_predicted_probs = None
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
    
  sgmcmc_train_epoch, evaluate = train_utils.make_sgd_train_epoch(
    net_apply, log_likelihood_fn, log_prior_fn, optimizer, num_batches)
  ensemble_acc = None
  
  assert train_set[0].dtype == dtype, (
      "Dataset data type {} does not match specified data type {}".format(
          train_set[0].dtype, dtype))
  assert jax.tree_flatten(params)[0][0].dtype == dtype, (
      "Params data type {} does not match specified data type {}".format(
          jax.tree_flatten(params)[0][0].dtype, dtype))
  
  for iteration in range(start_iteration, args.num_epochs):
    
    start_time = time.time()
    params, net_state, opt_state, logprob_avg, key = sgmcmc_train_epoch(
      params, net_state, opt_state, train_set, key)
    iteration_time = time.time() - start_time
    
    tabulate_dict = OrderedDict()
    tabulate_dict["iteration"] = iteration
    tabulate_dict["step_size"] = lr_schedule(opt_state.count)
    tabulate_dict["train_logprob"] = logprob_avg
    tabulate_dict["train_acc"] = None
    tabulate_dict["test_logprob"] = None
    tabulate_dict["test_acc"] = None
    tabulate_dict["ensemble_acc"] = ensemble_acc
    tabulate_dict["n_ens"] = num_ensembled
    tabulate_dict["time"] = iteration_time
    
    with tf_writer.as_default():
      tf.summary.scalar("train/log_prob_running", logprob_avg, step=iteration)
      tf.summary.scalar("hypers/step_size", lr_schedule(opt_state.count),
                        step=iteration)
      tf.summary.scalar("debug/iteration_time", iteration_time, step=iteration)
    
    if iteration % args.save_freq == 0 or iteration == args.num_epochs - 1:
      checkpoint_name = checkpoint_utils.make_checkpoint_name(iteration)
      checkpoint_path = os.path.join(dirname, checkpoint_name)
      checkpoint_dict = checkpoint_utils.make_sgmcmc_checkpoint_dict(
        iteration, params, net_state, opt_state, key, num_ensembled,
        ensemble_predicted_probs)
      checkpoint_utils.save_checkpoint(checkpoint_path, checkpoint_dict)
    
    if (iteration % args.eval_freq == 0) or (iteration == args.num_epochs - 1):
      test_log_prob, test_acc, test_ce, _ = evaluate(params, net_state,
                                                     test_set)
      train_log_prob, train_acc, train_ce, prior = (
        evaluate(params, net_state, train_set))

      tabulate_dict["train_logprob"] = train_log_prob
      tabulate_dict["test_logprob"] = test_log_prob
      tabulate_dict["train_acc"] = train_acc
      tabulate_dict["test_acc"] = test_acc
      with tf_writer.as_default():
        tf.summary.scalar("train/log_prob", train_log_prob, step=iteration)
        tf.summary.scalar("test/log_prob", test_log_prob, step=iteration)
        tf.summary.scalar("train/log_likelihood", train_ce, step=iteration)
        tf.summary.scalar("test/log_likelihood", test_ce, step=iteration)
        tf.summary.scalar("train/accuracy", train_acc, step=iteration)
        tf.summary.scalar("test/accuracy", test_acc, step=iteration)

    if ((iteration > args.num_burnin_epochs) and
        ((iteration - args.num_burnin_epochs) % args.ensemble_freq == 0)):
      ensemble_predicted_probs, ensemble_acc, num_ensembled = (
          train_utils.update_ensemble(
              net_apply, params, net_state, test_set, num_ensembled,
              ensemble_predicted_probs))
      tabulate_dict["ensemble_acc"] = ensemble_acc
      tabulate_dict["n_ens"] = num_ensembled
      test_labels = onp.asarray(test_set[1])
      
      ensemble_nll = metrics.nll(ensemble_predicted_probs, test_labels)
      ensemble_calibration = metrics.calibration_curve(
          ensemble_predicted_probs, test_labels)
      
      with tf_writer.as_default():
        tf.summary.scalar("test/ens_accuracy", ensemble_acc, step=iteration)
        tf.summary.scalar(
            "test/ens_ece", ensemble_calibration["ece"], step=iteration)
        tf.summary.scalar("test/ens_nll", ensemble_nll, step=iteration)
        tf.summary.scalar("debug/n_ens", num_ensembled, step=iteration)
    
    table = tabulate_utils.make_table(
      tabulate_dict, iteration - start_iteration, args.tabulate_freq)
    print(table)


if __name__ == "__main__":
  print("JAX sees the following devices:", jax.devices())
  print("TF sees the following devices:", tf.config.get_visible_devices())
  train_model()

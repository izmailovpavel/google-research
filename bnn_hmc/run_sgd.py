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

"""Run SGD training on a cloud TPU. We are not using data augmentation."""

import os
from jax import numpy as jnp
import numpy as onp
import jax
import tensorflow.compat.v2 as tf
import argparse
import time
from collections import OrderedDict

from bnn_hmc.core import data
from bnn_hmc.core import losses
from bnn_hmc.core import models
from bnn_hmc.utils import checkpoint_utils
from bnn_hmc.utils import cmd_args_utils
from bnn_hmc.utils import logging_utils
from bnn_hmc.utils import train_utils


parser = argparse.ArgumentParser(description="Run SGD on a cloud TPU")
cmd_args_utils.add_common_flags(parser)
parser.add_argument("--init_step_size", type=float, default=1.e-6,
                    help="Initial SGD step size")
parser.add_argument("--num_epochs", type=int, default=300,
                    help="Total number of SGD epochs iterations")
parser.add_argument("--batch_size", type=int, default=80, help="Batch size")
parser.add_argument("--momentum_decay", type=float, default=0.9,
                    help="Momentum decay parameter for SGD")
parser.add_argument("--eval_freq", type=int, default=10,
                    help="Frequency of evaluation (epochs)")
parser.add_argument("--save_freq", type=int, default=50,
                    help="Frequency of checkpointing (epochs)")

args = parser.parse_args()
train_utils.set_up_jax(args.tpu_ip, args.use_float64)


def train_model():
  subdirname = "sgd_wd_{}_stepsize_{}_batchsize_{}_momentum_{}_seed_{}".format(
    args.weight_decay, args.init_step_size, args.batch_size, args.momentum_decay,
    args.seed)
  dirname = os.path.join(args.dir, subdirname)
  os.makedirs(dirname, exist_ok=True)
  tf_writer = tf.summary.create_file_writer(dirname)
  cmd_args_utils.save_cmd(dirname, tf_writer)

  dtype = jnp.float64 if args.use_float64 else jnp.float32
  train_set, test_set, task, data_info = data.make_ds_pmap_fullbatch(
    args.dataset_name, dtype)
  
  net_apply, net_init = models.get_model(args.model_name, data_info)

  (likelihood_factory, predict_fn, ensemble_upd_fn, metrics_fns,
   tabulate_metrics) = train_utils.get_task_specific_fns(task, data_info)
  log_likelihood_fn = likelihood_factory(args.temperature)
  log_prior_fn, log_prior_diff_fn = losses.make_gaussian_log_prior(
    args.weight_decay, args.temperature)

  num_data = jnp.size(train_set[1])
  num_batches = num_data // args.batch_size
  num_devices = len(jax.devices())

  total_steps = num_batches * args.num_epochs
  lr_schedule = train_utils.make_cosine_lr_schedule(
      args.init_step_size, total_steps)
  optimizer = train_utils.make_optimizer(
      lr_schedule, momentum_decay=args.momentum_decay)
  
  checkpoint_dict, status = checkpoint_utils.initialize(
      dirname, args.init_checkpoint)
  if status == checkpoint_utils.InitStatus.INIT_RANDOM:
    key, net_init_key = jax.random.split(jax.random.PRNGKey(args.seed), 2)
    print("Starting from random initialization with provided seed")
    init_data = jax.tree_map(lambda elem: elem[0][:1], train_set)
    params, net_state = net_init(net_init_key, init_data, True)
    opt_state = optimizer.init(params)
    net_state = jax.pmap(lambda _: net_state)(jnp.arange(num_devices))
    key = jax.random.split(key, num_devices)
    start_iteration = 0
  else:
    start_iteration, params, net_state, opt_state, key = (
        checkpoint_utils.parse_sgd_checkpoint_dict(checkpoint_dict))
    if status == checkpoint_utils.InitStatus.INIT_CKPT:
      print("Resuming the run from the provided init_checkpoint")
      # TODO: fix -- we should only load the parameters in this case
    elif status == checkpoint_utils.InitStatus.LOADED_PREEMPTED:
      print("Continuing the run from the last saved checkpoint")

  sgd_train_epoch = train_utils.make_sgd_train_epoch(
    net_apply, log_likelihood_fn, log_prior_fn, optimizer, num_batches)
  
  for iteration in range(start_iteration, args.num_epochs):
    
    start_time = time.time()
    params, net_state, opt_state, logprob_avg, key = sgd_train_epoch(
        params, net_state, opt_state, train_set, key)
    iteration_time = time.time() - start_time
    
    if iteration % args.save_freq == 0 or iteration == args.num_epochs - 1:
      checkpoint_name = checkpoint_utils.make_checkpoint_name(iteration)
      checkpoint_path = os.path.join(dirname, checkpoint_name)
      checkpoint_dict = checkpoint_utils.make_sgd_checkpoint_dict(
          iteration, params, net_state, opt_state, key)
      checkpoint_utils.save_checkpoint(checkpoint_path, checkpoint_dict)

    # Evaluation
    train_stats = {"log_prob": logprob_avg}
    test_stats = {}

    if (iteration % args.eval_freq == 0) or (iteration == args.num_epochs - 1):
      test_predictions = onp.asarray(
        predict_fn(net_apply, params, net_state, test_set))
      train_predictions = onp.asarray(
        predict_fn(net_apply, params, net_state, train_set))
      test_stats = train_utils.evaluate_metrics(
        test_predictions, test_set[1], metrics_fns)
      train_stats = train_utils.evaluate_metrics(
        train_predictions, train_set[1], metrics_fns)
      train_stats["prior"] = log_prior_fn(params)

    # Logging
    other_logs = {
      "telemetry/iteration": iteration,
      "telemetry/iteration_time": iteration_time,
      "hypers/step_size": lr_schedule(opt_state[-1].count),
      "hypers/weight_decay": args.weight_decay,
      "hypers/temperature": args.temperature,
    }
    logging_dict = logging_utils.make_logging_dict(
      train_stats, test_stats, {})
    logging_dict.update(other_logs)

    with tf_writer.as_default():
      for stat_name, stat_val in logging_dict.items():
        tf.summary.scalar(stat_name, stat_val, step=iteration)
    tabulate_dict = OrderedDict()
    tabulate_dict["i"] = iteration
    tabulate_dict["t"] = iteration_time
    tabulate_dict["lr"] = lr_schedule(opt_state[-1].count)
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
  train_model()

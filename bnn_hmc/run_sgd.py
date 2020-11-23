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
import jax
import tensorflow.compat.v2 as tf
import argparse
import time
from collections import OrderedDict

from bnn_hmc import data
from bnn_hmc import models
from bnn_hmc import nn_loss
from bnn_hmc import train_utils
from bnn_hmc import checkpoint_utils
from bnn_hmc import cmd_args_utils
from bnn_hmc import tabulate_utils


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
train_utils.set_up_jax(args.tpu_ip)


def train_model():
  subdirname = "sgd_wd_{}_stepsize_{}_batchsize_{}_momentum_{}_seed_{}".format(
    args.weight_decay, args.init_step_size, args.batch_size, args.momentum_decay,
    args.seed)
  dirname = os.path.join(args.dir, subdirname)
  os.makedirs(dirname, exist_ok=True)
  tf_writer = tf.summary.create_file_writer(dirname)
  cmd_args_utils.save_cmd(dirname, tf_writer)

  dtype = jnp.float64 if args.use_float64 else jnp.float32
  train_set, test_set, num_classes = data.make_ds_pmap_fullbatch(
    args.dataset_name, dtype)
  
  net_apply, net_init = models.get_model(args.model_name, num_classes)
  
  log_likelihood_fn = nn_loss.make_xent_log_likelihood(num_classes, 1.)
  log_prior_fn, _ = (
    nn_loss.make_gaussian_log_prior(args.weight_decay, 1.))

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

  sgd_train_epoch, evaluate = train_utils.make_sgd_train_epoch(
    net_apply, log_likelihood_fn, log_prior_fn, optimizer, num_batches)
  
  for iteration in range(start_iteration, args.num_epochs):
    
    start_time = time.time()
    params, net_state, opt_state, logprob_avg, key = sgd_train_epoch(
        params, net_state, opt_state, train_set, key)
    iteration_time = time.time() - start_time

    tabulate_dict = OrderedDict()
    tabulate_dict["iteration"] = iteration
    tabulate_dict["step_size"] = lr_schedule(opt_state[-1].count)
    tabulate_dict["train_logprob"] = logprob_avg
    tabulate_dict["train_acc"] = None
    tabulate_dict["test_logprob"] = None
    tabulate_dict["test_acc"] = None
    tabulate_dict["time"] = iteration_time
    
    with tf_writer.as_default():
      tf.summary.scalar("train/log_prob_running", logprob_avg, step=iteration)
      tf.summary.scalar("hypers/step_size", lr_schedule(opt_state[-1].count),
                        step=iteration)
      tf.summary.scalar("debug/iteration_time", iteration_time, step=iteration)
    
    if iteration % args.save_freq == 0 or iteration == args.num_epochs - 1:
      checkpoint_name = checkpoint_utils.make_checkpoint_name(iteration)
      checkpoint_path = os.path.join(dirname, checkpoint_name)
      checkpoint_dict = checkpoint_utils.make_sgd_checkpoint_dict(
          iteration, params, net_state, opt_state, key)
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
    
    table = tabulate_utils.make_table(
        tabulate_dict, iteration - start_iteration, args.tabulate_freq)
    print(table)


if __name__ == "__main__":
  print("JAX sees the following devices:", jax.devices())
  train_model()

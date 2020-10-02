"""
Run SGD training on a cloud TPU. We are not using data augmentation.
"""

import os
import sys
from jax.config import config
import haiku as hk
from jax import numpy as jnp
import jax
import tensorflow.compat.v2 as tf
import argparse
import time
import tabulate
from collections import OrderedDict
from jax.experimental.callback import rewrite

from bnn_hmc import data
from bnn_hmc import models
from bnn_hmc import nn_loss
from bnn_hmc import train_utils
from bnn_hmc import precision_utils
from bnn_hmc import checkpoint_utils

parser = argparse.ArgumentParser(description="Run SGD on a cloud TPU")
parser.add_argument("--tpu_ip", type=str, default="10.0.0.2",
                    help="Cloud TPU internal ip "
                         "(see `gcloud compute tpus list`)")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--step_size", type=float, default=1.e-6,
                    help="Initial SGD step size")
parser.add_argument("--num_epochs", type=int, default=300,
                    help="Total number of SGD epochs iterations")
parser.add_argument("--batch_size", type=int, default=80, help="Batch size")
parser.add_argument("--weight_decay", type=float, default=15.,
                    help="Wight decay, equivalent to setting prior std")
parser.add_argument("--momentum_decay", type=float, default=0.9,
                    help="Momentum decay parameter for SGD")
parser.add_argument("--init_checkpoint", type=str, default=None,
                    help="Checkpoint to use for initialization of the chain")
parser.add_argument("--eval_freq", type=int, default=10,
                    help="Frequency of evaluation (epochs)")
parser.add_argument("--save_freq", type=int, default=50,
                    help="Frequency of checkpointing (epochs)")
parser.add_argument("--tabulate_freq", type=int, default=40,
                    help="Frequency of tabulate table header prints (epochs)")
parser.add_argument("--dir", type=str, default=None, required=True,
                    help="Directory for checkpoints and tensorboard logs")
parser.add_argument("--dataset_name", type=str, default="cifar10",
                    help="Name of the dataset")
parser.add_argument("--model_name", type=str, default="lenet",
                    help="Name of the dataset")

args = parser.parse_args()

config.FLAGS.jax_xla_backend = "tpu_driver"
config.FLAGS.jax_backend_target = "grpc://{}:8470".format(args.tpu_ip)

_MODEL_FNS = {"lenet": models.make_lenet_fn,
              "resnet18": models.make_resnet_18_fn}


def train_model():
  subdirname = "sgd_wd_{}_stepsize_{}_batchsize_{}_momentum_{}_seed_{}".format(
    args.weight_decay, args.step_size, args.batch_size, args.momentum_decay,
    args.seed)
  dirname = os.path.join(args.dir, subdirname)
  os.makedirs(dirname, exist_ok=True)
  with open(os.path.join(dirname, "comand.sh"), "w") as f:
    f.write(" ".join(sys.argv))
    f.write("\n")
  
  tf_writer = tf.summary.create_file_writer(dirname)
  
  train_set, test_set, num_classes = data.make_ds_pmap_fullbatch(
    name=args.dataset_name)
  net_fn = _MODEL_FNS[args.model_name](num_classes)
  net = hk.transform_with_state(net_fn)
  net_apply = net.apply
  net_apply = jax.experimental.callback.rewrite(
    net_apply,
    precision_utils.HIGH_PRECISION_RULES)
  
  log_likelihood_fn = nn_loss.xent_log_likelihood
  log_prior_fn, log_prior_diff = (
    nn_loss.make_gaussian_log_prior(weight_decay=args.weight_decay))

  loss_grad_fn = train_utils.make_log_prob_and_grad_nopmap_fn(
      net_apply, log_likelihood_fn, log_prior_fn)
  eval_fn = train_utils.make_log_prob_and_acc_fn(
      net_apply, log_likelihood_fn, log_prior_fn)

  num_data = jnp.size(train_set[1])
  num_batches = num_data // args.batch_size
  num_devices = len(jax.devices())

  total_steps = num_batches * args.num_epochs
  lr_schedule = train_utils.make_cosine_lr_schedule(args.step_size, total_steps)
  optimizer = train_utils.make_optimizer(
      lr_schedule, momentum_decay=args.momentum_decay)
  
  checkpoint_dict, status = checkpoint_utils.initialize(
      dirname, args.init_checkpoint)
  if status == checkpoint_utils.InitStatus.INIT_RANDOM:
    key, net_init_key = jax.random.split(jax.random.PRNGKey(args.seed), 2)
    print("Starting from random initialization with provided seed")
    init_data = jax.tree_map(lambda elem: elem[0][:1], train_set)
    params, net_state = net.init(net_init_key, init_data, True)
    opt_state = optimizer.init(params)
    net_state = jax.pmap(lambda _: net_state)(jnp.arange(num_devices))
    key = jax.random.split(jax.random.PRNGKey(0), num_devices)
    start_iteration = 0
  else:
    start_iteration, params, net_state, opt_state, key = (
        checkpoint_utils.parse_sgd_checkpoint_dict(checkpoint_dict))
    if status == checkpoint_utils.InitStatus.INIT_CKPT:
      print("Resuming the run from the provided init_checkpoint")
    elif status == checkpoint_utils.InitStatus.LOADED_PREEMPTED:
      print("Continuing the run from the last saved checkpoint")

  sgd_train_epoch = train_utils.make_sgd_train_epoch(
      loss_grad_fn, optimizer, num_batches)
  
  for iteration in range(start_iteration, args.num_epochs):
    
    start_time = time.time()
    logprobs, params, net_state, opt_state, key = sgd_train_epoch(
        params, net_state, opt_state, train_set, key)
    params = jax.tree_map(lambda p: p[0], params)
    opt_state = jax.tree_map(lambda p: p[0], opt_state)
    iteration_time = time.time() - start_time
    
    logprob_avg = jnp.mean(logprobs)

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
    
    if iteration % args.eval_freq == 0:
      test_log_prob, test_acc = eval_fn(params, net_state, test_set)
      train_log_prob, train_acc = eval_fn(params, net_state, test_set)
      
      tabulate_dict["train_logprob"] = train_log_prob
      tabulate_dict["test_logprob"] = test_log_prob
      tabulate_dict["train_acc"] = train_acc
      tabulate_dict["test_acc"] = test_acc
      with tf_writer.as_default():
        tf.summary.scalar("train/log_prob", train_log_prob, step=iteration)
        tf.summary.scalar("test/log_prob", test_log_prob, step=iteration)
        tf.summary.scalar("train/accuracy", train_acc, step=iteration)
        tf.summary.scalar("test/accuracy", test_acc, step=iteration)
    
    table = tabulate.tabulate([tabulate_dict.values()],
                              tabulate_dict.keys(),
                              tablefmt='simple', floatfmt='8.7f')
    if (iteration - start_iteration) % args.tabulate_freq == 0:
      table = table.split('\n')
      table = '\n'.join([table[1]] + table)
    else:
      table = table.split('\n')[2]
    print(table)


if __name__ == "__main__":
  print("JAX sees the following devices:", jax.devices())
  train_model()

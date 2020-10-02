"""Run an Hamiltonian Monte Carlo chain on a cloud TPU."""

import os
import sys
from jax.config import config
import haiku as hk
import numpy as onp
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

parser = argparse.ArgumentParser(description="Run an HMC chain on a cloud TPU")
parser.add_argument("--tpu_ip", type=str, default="10.0.0.2",
                    help="Cloud TPU internal ip "
                         "(see `gcloud compute tpus list`)")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
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
parser.add_argument("--num_burn_in_iterations", type=int, default=0,
                    help="Number of burn-in iterations")
parser.add_argument("--weight_decay", type=float, default=15.,
                    help="Wight decay, equivalent to setting prior std")
parser.add_argument("--init_checkpoint", type=str, default=None,
                    help="Checkpoint to use for initialization of the chain")
parser.add_argument("--tabulate_freq", type=int, default=40,
                    help="Frequency of tabulate table header prints")
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
  subdirname = "wd_{}_stepsize_{}_trajlen_{}_seed_{}_burnin_{}_{}".format(
      args.weight_decay, args.step_size, args.trajectory_len,
      args.seed, args.num_burn_in_iterations, args.burn_in_step_size_factor)
  dirname = os.path.join(args.dir, subdirname)
  os.makedirs(dirname, exist_ok=True)
  with open(os.path.join(dirname, "comand.sh"), "w") as f:
    f.write(" ".join(sys.argv))
    f.write("\n")

  tf_writer = tf.summary.create_file_writer(dirname)
  
  train_set, test_set, num_classes = data.make_ds_pmap_fullbatch(name=args.dataset_name)
  net_fn = _MODEL_FNS[args.model_name](num_classes)
  net = hk.transform_with_state(net_fn)
  net_apply = net.apply
  net_apply = jax.experimental.callback.rewrite(
    net_apply,
    precision_utils.HIGH_PRECISION_RULES)
  
  log_likelihood_fn = nn_loss.xent_log_likelihood
  log_prior_fn, log_prior_diff = (
      nn_loss.make_gaussian_log_prior(weight_decay=args.weight_decay))

  log_prob_and_grad_fn, log_prob_and_acc_fn = (
      train_utils.make_log_prob_grad_and_eval_fns(
          net_apply, log_likelihood_fn, log_prior_fn, train_set))
  update_fn = train_utils.make_hmc_update_eval_fns(
      log_prob_and_grad_fn,
      log_prior_diff, args.target_accept_rate,
      args.step_size_adaptation_speed)
  
  eval_fn = train_utils.make_evaluate_fn(
      log_prob_and_acc_fn, [train_set, test_set])

  trajectory_len = args.trajectory_len
  
  checkpoints = filter(train_utils.name_is_checkpoint, os.listdir(dirname))
  checkpoints = list(checkpoints)
  if checkpoints:
    print("Continuing the run from the last saved checkpoint")
    checkpoint_iteration = map(train_utils.parse_checkpoint_name, checkpoints)
    start_iteration = max(checkpoint_iteration)
    start_checkpoint_path = (
        os.path.join(dirname, train_utils.make_checkpoint_name(start_iteration)))
    (params, net_state, key, step_size, _, n_ensembled,
     ensemble_predicted_probs) = (
        train_utils.load_checkpoint(start_checkpoint_path))

  else:
    key, net_init_key = jax.random.split(jax.random.PRNGKey(args.seed), 2)
    step_size = args.step_size
    start_iteration = 0

    if args.init_checkpoint is not None:
      print("Resuming the run from the provided init_checkpoint")
      params, net_state, _, _ = (
          train_utils.load_checkpoint(args.init_checkpoint))
      n_ensembled = 0
      ensemble_predicted_probs = None

    else:
      print("Starting from random initialization with provided seed")
      init_data = jax.tree_map(lambda elem: elem[0][:1], train_set)
      params, net_state = net.init(net_init_key, init_data, True)
      n_devices = len(jax.devices())
      net_state = jax.pmap(lambda _: net_state)(jnp.arange(n_devices))
      n_ensembled = 0
      ensemble_predicted_probs = None

  log_prob, state_grad, log_likelihood, _, net_state = (
      log_prob_and_grad_fn(params, net_state))

  ensemble_acc = 0
  
  for iteration in range(start_iteration, args.num_iterations):
    
    # do a linear ramp-down of the step-size in the burn-in phase
    if iteration < args.num_burn_in_iterations:
      alpha = iteration / (args.num_burn_in_iterations - 1)
      initial_step_size = args.step_size
      final_step_size = args.burn_in_step_size_factor * args.step_size
      step_size = final_step_size * alpha + initial_step_size * (1 - alpha)
      
    start_time = time.time()
    do_mh_correction = (iteration >= args.num_burn_in_iterations)
    (params, net_state, log_likelihood, state_grad, step_size, key,
     accept_prob, accepted) = (
        update_fn(params, net_state, log_likelihood, state_grad,
                  key, step_size, trajectory_len, do_mh_correction)
    )
    iteration_time = time.time() - start_time

    checkpoint_name = train_utils.make_checkpoint_name(iteration)
    checkpoint_path = os.path.join(dirname, checkpoint_name)
    checkpoint_dict = train_utils.make_checkpoint_dict(
      params, net_state, key, step_size, accepted, n_ensembled,
      ensemble_predicted_probs)
    train_utils.save_checkpoint(checkpoint_path, checkpoint_dict)

    if do_mh_correction and accepted:
      predicted_probs = nn_loss.pmap_get_softmax_predictions(
        net_apply, params, net_state, test_set, 1, False)
      predicted_probs = onp.asarray(predicted_probs)
      if n_ensembled:
        ensemble_predicted_probs += (
            (predicted_probs - ensemble_predicted_probs) / (n_ensembled + 1))
      else:
        ensemble_predicted_probs = predicted_probs
      n_ensembled += 1
      ensemble_preds = onp.argmax(ensemble_predicted_probs, -1)[:, 0]
      ensemble_acc = (ensemble_preds == test_set[1]).mean()

    (test_log_prob, test_acc), (train_log_prob, train_acc) = (
      eval_fn(params, net_state))
      
    tabulate_dict = OrderedDict()
    tabulate_dict["iteration"] = iteration
    tabulate_dict["step_size"] = step_size
    tabulate_dict["train_logprob"] = log_prob
    tabulate_dict["test_logprob"] = test_log_prob
    tabulate_dict["train_acc"] = train_acc
    tabulate_dict["test_acc"] = test_acc
    tabulate_dict["accept_prob"] = accept_prob
    tabulate_dict["accepted"] = accepted
    tabulate_dict["ensemble_acc"] = ensemble_acc
    tabulate_dict["n_ens"] = n_ensembled
    tabulate_dict["time"] = iteration_time

    with tf_writer.as_default():
      tf.summary.scalar("train/log_likelihood", log_likelihood, step=iteration)
      tf.summary.scalar("train/log_prob", train_log_prob, step=iteration)
      tf.summary.scalar("test/log_prob", test_log_prob, step=iteration)
      tf.summary.scalar("train/accuracy", train_acc, step=iteration)
      tf.summary.scalar("test/accuracy", test_acc, step=iteration)
      tf.summary.scalar("test/ens_accuracy", ensemble_acc, step=iteration)
      tf.summary.scalar("hypers/step_size", step_size, step=iteration)
      tf.summary.scalar("hypers/trajectory_len", trajectory_len,
                        step=iteration)
      tf.summary.scalar("debug/accept_prob", accept_prob, step=iteration)
      tf.summary.scalar("debug/do_mh_correction", float(do_mh_correction),
                        step=iteration)
      tf.summary.scalar("debug/iteration_time", iteration_time, step=iteration)
      tf.summary.scalar("debug/accepted", accepted, step=iteration)
      tf.summary.scalar("debug/n_ens", n_ensembled, step=iteration)

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

"""Run an Hamiltonian Monte Carlo chain on a cloud TPU."""

import os
import sys
from jax.config import config
import haiku as hk
import jax
from jax import numpy as jnp
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
from bnn_hmc import tree_utils
from bnn_hmc import precision_utils


parser = argparse.ArgumentParser(description="Run an HMC chain on a cloud TPU")
parser.add_argument("--tpu_ip", type=str, default="10.0.0.2",
                    help="Cloud TPU internal ip "
                         "(see `gcloud compute tpus list`)")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--weight_decay", type=float, default=15.,
                    help="Wight decay, equivalent to setting prior std")
parser.add_argument("--dataset_name", type=str, default="cifar10",
                    help="Name of the dataset")
parser.add_argument("--model_name", type=str, default="lenet",
                    help="Name of the dataset")

args = parser.parse_args()

config.FLAGS.jax_xla_backend = "tpu_driver"
config.FLAGS.jax_backend_target = "grpc://{}:8470".format(args.tpu_ip)

_MODEL_FNS = {"lenet": models.lenet_fn}



def make_gaussian_prior_difference_numpy(weight_decay):
  """Returns the function that computes the difference in prior."""
  def prior_diff(params1, params2):
    """Computes the delta in  Gaussian prior negative log-density."""
    diff = sum([onp.sum(onp.array(p1)**2 - onp.array(p2)**2) for p1, p2 in
                zip(jax.tree_leaves(params1), jax.tree_leaves(params2))])
    return (0.5 * weight_decay * diff) 

  return prior_diff


def test_precision():

    net_fn = _MODEL_FNS[args.model_name]
    net_fn_highprec = jax.experimental.callback.rewrite(
  			net_fn,
  			precision_utils.HIGH_PRECISION_RULES)
    net = hk.transform(net_fn)
    net_highprec = hk.transform(net_fn_highprec)

    train_set, test_set, _ = data.make_ds_pmap_fullbatch(name=args.dataset_name)
    likelihood_fn = nn_loss.xent_likelihood
    prior_fn = nn_loss.make_gaussian_prior(weight_decay=args.weight_decay)
    prior_diff_fn = nn_loss.make_gaussian_prior_difference(weight_decay=args.weight_decay)
    prior_diff_np_fn = make_gaussian_prior_difference_numpy(weight_decay=args.weight_decay)

    #update_fn, eval_fn, log_prob_and_grad_fn = (
    #    train_utils.make_hmc_update_eval_fns(net, train_set, test_set,
    #                                         likelihood_fn, prior_fn))

    key, net_init_key, net_init_key2 = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    init_data = jax.tree_map(lambda elem: elem[0], train_set)
    params = net.init(net_init_key, init_data)
    params2 = net.init(net_init_key2, init_data)
    
    #log_prob, state_grad = log_prob_and_grad_fn(params)
    tabulate_columns = ["iteration", "train_logprob", "train_acc", "test_logprob",
                        "test_acc", "step_size", "accept_prob", "time"]

    n_devices = len(jax.local_devices())
    params_p = jax.pmap(lambda _: params)(jnp.arange(n_devices))
    params2_p = jax.pmap(lambda _: params2)(jnp.arange(n_devices))

    preds = nn_loss.pmap_get_softmax_preds(net, params_p, test_set, 1)
    preds_highprec = nn_loss.pmap_get_softmax_preds(net_highprec, params_p, test_set, 1)
    print("Distance between predictions:", jnp.sum((preds - preds_highprec)**2))

    prior_val = prior_fn(params_p)
    prior_val2 = prior_fn(params2_p)
    prior_diff = prior_val - prior_val2
    prior_diff_highprec = prior_diff_fn(params_p, params2_p)
    prior_diff_np = prior_diff_np_fn(params_p, params2_p)
    print("Prior diff", prior_diff, prior_diff_highprec, prior_diff_np)
    print(prior_diff_np.dtype)



if __name__ == "__main__":
    print("JAX sees the following devices:", jax.devices())
    test_precision()

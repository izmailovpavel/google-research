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
    prior_fn_highprec = jax.experimental.callback.rewrite(
  			  prior_fn,
  			  precision_utils.HIGH_PRECISION_RULES)

    #update_fn, eval_fn, log_prob_and_grad_fn = (
    #    train_utils.make_hmc_update_eval_fns(net, train_set, test_set,
    #                                         likelihood_fn, prior_fn))

    key, net_init_key = jax.random.split(jax.random.PRNGKey(args.seed), 2)
    init_data = jax.tree_map(lambda elem: elem[0], train_set)
    params = net.init(net_init_key, init_data)
    
    #log_prob, state_grad = log_prob_and_grad_fn(params)
    tabulate_columns = ["iteration", "train_logprob", "train_acc", "test_logprob",
                        "test_acc", "step_size", "accept_prob", "time"]

    n_devices = len(jax.local_devices())
    start = time.time()
    params_p = jax.pmap(lambda _: params)(jnp.arange(n_devices))
    print(time.time() - start)
    start = time.time()
    params_p = jax.pmap(lambda _: params)(jnp.arange(n_devices))
    print(time.time() - start)
    preds = nn_loss.pmap_get_softmax_preds(net, params_p, test_set, 1)
    preds_highprec = nn_loss.pmap_get_softmax_preds(net_highprec, params_p, test_set, 1)
    print("Distance between predictions:", jnp.sum((preds - preds_highprec)**2))

    prior_val = prior_fn(params_p)
    prior_highprec_val = prior_fn_highprec(params_p)
    print("Prior", prior_val, prior_highprec_val)



if __name__ == "__main__":
    print("JAX sees the following devices:", jax.devices())
    test_precision()

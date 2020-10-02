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

"""Loss computation for nn."""
import functools
import math
from typing import Callable, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp

from bnn_hmc import tree_utils


# Batch = Tuple[onp.ndarray, onp.ndarray]
# LossAcc = Tuple[jnp.ndarray, jnp.ndarray]
# LossAccGrad = Tuple[jnp.ndarray, jnp.ndarray, hk.Params]
# PriorFn = Callable[[hk.Params], jnp.array]
# LikelihoodFn = Callable[[hk.Transformed, hk.Params, Batch], LossAcc]


def xent_log_likelihood(net_apply, params, net_state, batch, is_training):
  """Computes the negative log-likelihood."""
  _, y = batch
  logits, net_state = net_apply(params, net_state, None, batch, is_training)
  labels = jax.nn.one_hot(y, 10)
  softmax_xent = jnp.sum(labels * jax.nn.log_softmax(logits))

  accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
  return softmax_xent, (accuracy, net_state)


def make_gaussian_log_prior(weight_decay):
  """Returns the Gaussian log-density and delta given weight decay."""
  def log_prior(params):
    """Computes the Gaussian prior log-density."""
    n_params = sum([p.size for p in jax.tree_leaves(params)])
    return -(0.5 * tree_utils.tree_dot(params, params) * weight_decay +
             0.5 * n_params * jnp.log(weight_decay / (2 * math.pi)))
  
  def log_prior_diff(params1, params2):
    """Computes the delta in  Gaussian prior log-density."""
    diff = sum([jnp.sum(p1**2 - p2**2) for p1, p2 in
                zip(jax.tree_leaves(params1), jax.tree_leaves(params2))])
    return -0.5 * weight_decay * diff

  return log_prior, log_prior_diff


@functools.partial(
    jax.pmap, axis_name='i', static_broadcasted_argnums=[0, 3],
    in_axes=(None, None, 0, None, 0)
)
def pmap_get_log_likelihood_and_grad(
    net_apply, params, net_state, likelihood_fn, dataset
):
  loss_acc_val_grad = jax.value_and_grad(likelihood_fn, has_aux=True,
                                         argnums=1)
  (likelihood, (_, net_state)), likelihood_grad = (
      loss_acc_val_grad(net_apply, params, net_state, dataset, is_training=True))
  likelihood = jax.lax.psum(likelihood, axis_name='i')
  likelihood_grad = jax.lax.psum(likelihood_grad, axis_name='i')
  return likelihood, likelihood_grad, net_state


@functools.partial(
    jax.pmap, axis_name='i', static_broadcasted_argnums=[0, 3, 4],
    in_axes=(None, None, 0, None, None, 0)
)
def pmap_get_log_prob_and_accuracy(
    net_apply, params, net_state, likelihood_fn, prior_fn, dataset, is_training=False
):
  """Computes posterior density value and accuracy via pmap."""

  likelihood, (acc, net_state) = likelihood_fn(
      net_apply, params, net_state, dataset, is_training)
  prior = prior_fn(params)

  acc = jax.lax.pmean(acc, axis_name='i')
  likelihood = jax.lax.psum(likelihood, axis_name='i')
  prior = jax.lax.pmean(prior, axis_name='i')

  return likelihood + prior, acc, net_state


@functools.partial(
    jax.pmap, axis_name='i', static_broadcasted_argnums=[0, 4, 5],
    in_axes=(None, None, 0, 0, None, None)
)
def pmap_get_softmax_predictions(
    net_apply, params, net_state, dataset, num_batches, is_training=False
):
  """Computes predictions via pmap."""

  batch_size = dataset[0].shape[0] // num_batches
  dataset = jax.tree_map(
      lambda x: x.reshape((num_batches, batch_size, *x.shape[1:])), dataset)

  def get_batch_predictions(_, x):
    y, _ = net_apply(params, net_state, None, x, is_training)
    batch_predictions = jax.nn.softmax(y)
    return None, batch_predictions

  _, predictions = jax.lax.scan(get_batch_predictions, None, dataset)

  return predictions

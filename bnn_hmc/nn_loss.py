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

"""Likelihood and prior functions."""
import math
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from bnn_hmc import tree_utils


# Batch = Tuple[onp.ndarray, onp.ndarray]
# LossAcc = Tuple[jnp.ndarray, jnp.ndarray]
# LossAccGrad = Tuple[jnp.ndarray, jnp.ndarray, hk.Params]
# PriorFn = Callable[[hk.Params], jnp.array]
# LikelihoodFn = Callable[[hk.Transformed, hk.Params, Batch], LossAcc]

def make_xent_log_likelihood(num_classes):
  def xent_log_likelihood(net_apply, params, net_state, batch, is_training):
    """Computes the negative log-likelihood."""
    _, y = batch
    logits, net_state = net_apply(params, net_state, None, batch, is_training)
    labels = jax.nn.one_hot(y, num_classes)
    softmax_xent = jnp.sum(labels * jax.nn.log_softmax(logits))
  
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return softmax_xent, (accuracy, net_state)
  return xent_log_likelihood


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


def make_gaussian_likelihood(noise_std=None):
  def gaussian_log_likelihood(net_apply, params, net_state, batch, is_training):
    """Computes the negative log-likelihood."""
    _, y = batch
    predictions, net_state = net_apply(
        params, net_state, None, batch, is_training)
    
    if noise_std is None:
      predictions, predictions_std = jnp.split(predictions, [1], axis=-1)
    else:
      predictions_std = noise_std
      
    mse = (predictions - y)**2
    mse /= predictions_std**2
    log_likelihood = -jnp.sum(mse)
    
    statistics = jnp.mean(mse)
    
    return log_likelihood, (statistics, net_state)
  return gaussian_log_likelihood

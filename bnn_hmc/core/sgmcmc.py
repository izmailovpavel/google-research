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

"""Optax implementations of SGMCMC optimizers."""

import jax
from optax import OptState
from jax import numpy as jnp
from optax import GradientTransformation
from typing import Any

from bnn_hmc.utils import tree_utils


Momentum = Any


class OptaxSGLDState(OptState):
  """Optax state for the SGLD optimizer"""
  count: jnp.ndarray
  rng_key: jnp.ndarray
  
  
def sgld_gradient_update(step_size_fn, seed):
  """Optax implementation of the SGLD optimizer"""

  def init_fn(_):
    return OptaxSGLDState(count=jnp.zeros([], jnp.int32),
                          rng_key=jax.random.PRNGKey(seed))
  
  def update_fn(updates, state, params=None):
    del params
    lr = step_size_fn(state.count)
    noise_std = jnp.sqrt(2 / lr)

    # add noise to gradients
    noise, new_key = tree_utils.normal_like_tree(updates, state.rng_key)
    updates = jax.tree_multimap(lambda g, n: g + noise_std * n, updates, noise)
    
    # apply lr schedule
    updates = jax.tree_map(lambda g: g * lr, updates)
    return updates, OptaxSGLDState(
        count=state.count + 1, rng_key=new_key)
  
  return GradientTransformation(init_fn, update_fn)


class OptaxSGHMCState(OptState):
  """Optax state for the SGHMC optimizer"""
  count: jnp.ndarray
  rng_key: jnp.ndarray
  momentum: Momentum


def sghmc_gradient_update(step_size_fn, momentum_decay, seed):
  """Optax implementation of the SGHMC optimizer"""

  def init_fn(params):
    return OptaxSGHMCState(count=jnp.zeros([], jnp.int32),
                           rng_key=jax.random.PRNGKey(seed),
                           momentum=jax.tree_map(jnp.zeros_like, params))

  def update_fn(updates, state, params=None):
    del params
    lr = step_size_fn(state.count)
    noise_std = jnp.sqrt(2 * lr * (1 - momentum_decay))

    noise, new_key = tree_utils.normal_like_tree(updates, state.rng_key)

    def update_momentum(m, g, n):
      return momentum_decay * m + g * lr + n * noise_std

    momentum = jax.tree_multimap(
        update_momentum, state.momentum, updates, noise)

    return momentum, OptaxSGHMCState(
        count=state.count + 1, rng_key=new_key, momentum=momentum)

  return GradientTransformation(init_fn, update_fn)


def get_sgmcmc_optimizer(lr_schedule, args):
  method_name = args.method_name

  if method_name.lower() == "sgld":
    return sgld_gradient_update(lr_schedule, args.seed)
  elif method_name.lower() == "sghmc":
    return sghmc_gradient_update(lr_schedule, args.sghmc_momentum, args.seed)
  else:
    raise ValueError("Unknown SG-MCMC method {}".format(method_name))
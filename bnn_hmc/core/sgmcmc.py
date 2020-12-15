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

"""Optix implementations of SGMCMC optimizers."""

import jax
from jax.experimental.optix import OptState
from jax import numpy as jnp
from jax.experimental.optix import GradientTransformation

from bnn_hmc.utils import tree_utils


class OptixSGLDState(OptState):
  """Optix state for the SGLD optimizer"""
  count: jnp.ndarray
  rng_key: jnp.ndarray
  
  
def sgld_gradient_update(step_size_fn, seed):
  """Optix implementation of the SGLD optimizer"""

  def init_fn(_):
    return OptixSGLDState(count=jnp.zeros([], jnp.int32),
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
    return updates, OptixSGLDState(
        count=state.count + 1, rng_key=new_key)
  
  return GradientTransformation(init_fn, update_fn)
  
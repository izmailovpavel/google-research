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

"""Utility functions for DNN training."""

from typing import Callable

import jax
from jax.experimental import optix
import jax.numpy as jnp
import numpy as onp
import functools
import os

from bnn_hmc import hmc
from bnn_hmc import nn_loss


LRSchedule = Callable[[jnp.ndarray], jnp.ndarray]
Opt = optix.GradientTransformation


def make_cosine_lr_schedule(init_lr, total_steps):
  """Cosine LR schedule."""
  def schedule(step):
    t = step / total_steps
    return 0.5 * init_lr * (1 + jnp.cos(t * onp.pi))
  return schedule


def make_optimizer(lr_schedule, momentum_decay):
  # Maximize log-prob instead of minimizing loss
  return optix.chain(optix.trace(decay=momentum_decay, nesterov=False),
                     optix.scale_by_schedule(lr_schedule))


def make_log_prob_and_grad_fn(
    net_apply, log_likelihood_fn, log_prior_fn
):
  def log_prob_and_grad_fn(params, net_state, dataset):
    likelihood, likelihood_grad, net_state = (
        nn_loss.pmap_get_log_likelihood_and_grad(
            net_apply, params, net_state, log_likelihood_fn, dataset))
    prior, prior_grad = jax.value_and_grad(log_prior_fn)(params)
    log_prob = likelihood[0] + prior
    grad = jax.tree_multimap(lambda g_l, g_p: g_l[0] + g_p,
                             likelihood_grad, prior_grad)
    return log_prob, grad, likelihood[0], prior, net_state

  return log_prob_and_grad_fn


def make_log_prob_and_grad_nopmap_fn(
    net_apply, log_likelihood_fn, log_prior_fn
):
  def log_prob_and_grad_fn(params, net_state, batch, total_num_data):
    loss_acc_val_grad = jax.value_and_grad(log_likelihood_fn, has_aux=True,
                                           argnums=1)
    (likelihood, (_, net_state)), likelihood_grad = (
      loss_acc_val_grad(net_apply, params, net_state, batch, is_training=True)
    )
    prior, prior_grad = jax.value_and_grad(log_prior_fn)(params)
    coefficient = total_num_data / jnp.size(batch[1])
    log_prob = coefficient * likelihood + prior
    grad = jax.tree_multimap(lambda g_l, g_p: coefficient * g_l + g_p,
                             likelihood_grad, prior_grad)
    return log_prob, grad, likelihood, prior, net_state

  return log_prob_and_grad_fn


def make_log_prob_and_acc_fn(
    net_apply, log_likelihood_fn, log_prior_fn
):
  def log_prob_and_acc_fn(params, net_state, dataset):
    log_prob, acc, _ = nn_loss.pmap_get_log_prob_and_accuracy(
      net_apply, params, net_state, log_likelihood_fn, log_prior_fn, dataset)
    return log_prob[0], acc[0]
  
  return log_prob_and_acc_fn


def make_hmc_update_eval_fns(
    log_prob_and_grad_fn, log_prior_diff_fn, target_accept_rate=0.9,
    step_size_adaptation_speed=0.
):
  """Make update and ev0al functions for HMC training."""

  hmc_update = hmc.make_adaptive_hmc_update(
      log_prob_and_grad_fn, log_prior_diff_fn)

  def update(
      params, net_state, log_likelihood, state_grad, key, step_size,
      trajectory_len, do_mh_correction
  ):
    (params, net_state, log_likelihood, state_grad, step_size, accept_prob,
     accepted) = (
        hmc_update(
            params, net_state, log_likelihood, state_grad, key, step_size,
            trajectory_len, target_accept_rate=target_accept_rate,
            step_size_adaptation_speed=step_size_adaptation_speed,
            do_mh_correction=do_mh_correction))
    key, = jax.random.split(key, 1)
    return (params, net_state, log_likelihood, state_grad, step_size, key,
            accept_prob, accepted)

  return update, log_prob_and_grad_fn


def make_sgd_train_epoch(loss_grad_fn, optimizer, num_batches):
  """
  Make a training epoch function for SGD-like optimizers.
  """
  
  @functools.partial(
    jax.pmap, axis_name='i', static_broadcasted_argnums=[],
    in_axes=(None, 0, None, 0, 0)
  )
  def sgd_train_epoch(params, net_state, opt_state, train_set, key):
    n_data = train_set[0].shape[0]
    batch_size = n_data // num_batches
    indices = jax.random.permutation(key, jnp.arange(n_data))
    indices = jax.tree_map(
        lambda x: x.reshape((num_batches, batch_size)), indices)

    total_num_data = jax.lax.psum(jnp.size(train_set[1]), axis_name='i')
    
    def train_step(carry, batch_indices):
      batch = jax.tree_map(lambda x: x[batch_indices], train_set)
      params_, net_state_, opt_state_ = carry
      loss, grad, _, _, net_state_ = loss_grad_fn(
        params_, net_state_, batch, total_num_data)
      grad = jax.lax.psum(grad, axis_name='i')
      
      updates, opt_state_ = optimizer.update(grad, opt_state_)
      params_ = optix.apply_updates(params_, updates)
      return (params_, net_state_, opt_state_), loss
    
    (params, net_state, opt_state), losses = jax.lax.scan(
        train_step, (params, net_state, opt_state), indices)
    
    new_key, = jax.random.split(key, 1)
    return losses, params, net_state, opt_state, new_key
  return sgd_train_epoch

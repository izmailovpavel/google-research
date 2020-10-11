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
from jax.config import config

from bnn_hmc import hmc
from bnn_hmc import tree_utils


LRSchedule = Callable[[jnp.ndarray], jnp.ndarray]
Opt = optix.GradientTransformation
_CHECKPOINT_FORMAT_STRING = "model_step_{}.pt"


def set_up_jax(tpu_ip):
  if tpu_ip is not None:
    config.FLAGS.jax_xla_backend = "tpu_driver"
    config.FLAGS.jax_backend_target = "grpc://{}:8470".format(tpu_ip)


def make_cosine_lr_schedule(init_lr, total_steps):
  """Cosine LR schedule."""
  def schedule(step):
    t = step / total_steps
    return 0.5 * init_lr * (1 + jnp.cos(t * onp.pi))
  return schedule


def make_optimizer(lr_schedule, momentum_decay):
  """Make SGD optimizer with momentum."""
  # Maximize log-prob instead of minimizing loss
  return optix.chain(optix.trace(decay=momentum_decay, nesterov=False),
                     optix.scale_by_schedule(lr_schedule))


def make_perdevice_log_prob_acc_grad_fns(
    net_apply, log_likelihood_fn, log_prior_fn
):
  """Functions for training and evaluation, should be jax.lax.psum'ed"""
  def likelihood_prior_and_grads_fn(params, net_state, batch):
    loss_acc_val_grad = jax.value_and_grad(
        log_likelihood_fn, has_aux=True, argnums=1)
    (likelihood, (acc, net_state)), likelihood_grad = loss_acc_val_grad(
        net_apply, params, net_state, batch, is_training=True)
    prior, prior_grad = jax.value_and_grad(log_prior_fn)(params)
    return likelihood, likelihood_grad, prior, prior_grad, acc, net_state
  
  def likelihood_prior_and_acc_fn(params, net_state, batch, is_training):
    likelihood, (acc, net_state) = log_likelihood_fn(
        net_apply, params, net_state, batch, is_training=is_training)
    prior = log_prior_fn(params)
    return likelihood, prior, acc, net_state

  return likelihood_prior_and_grads_fn, likelihood_prior_and_acc_fn


def make_eval_fn(likelihood_prior_and_acc_fn):
  """Define evaluation function."""
  @functools.partial(
    jax.pmap, axis_name='i', in_axes=(None, 0, 0)
  )
  def pmap_eval(params, net_state, dataset):
    likelihood, prior, acc, _ = likelihood_prior_and_acc_fn(
      params, net_state, dataset, is_training=False)
    likelihood = jax.lax.psum(likelihood, axis_name='i')
    log_prob = likelihood + prior
    acc = jax.lax.pmean(acc, axis_name='i')
    return log_prob, acc, likelihood, prior
  
  def evaluate(params, net_state, dataset):
    return (arr[0] for arr in pmap_eval(params, net_state, dataset))
  
  return evaluate


def make_hmc_update(
    net_apply, log_likelihood_fn, log_prior_fn,
    log_prior_diff_fn, target_accept_rate=0.9, step_size_adaptation_speed=0.
):
  """Make update and ev0al functions for HMC training."""

  perdevice_likelihood_prior_and_grads_fn, likelihood_prior_and_acc_fn = (
      make_perdevice_log_prob_acc_grad_fns(
          net_apply, log_likelihood_fn, log_prior_fn))
  
  def _perdevice_log_prob_and_grad(dataset, params, net_state):
    # Only call inside pmap
    likelihood, likelihood_grad, prior, prior_grad, _, net_state = (
        perdevice_likelihood_prior_and_grads_fn(params, net_state, dataset))
    likelihood = jax.lax.psum(likelihood, axis_name='i')
    likelihood_grad = jax.lax.psum(likelihood_grad, axis_name='i')
    log_prob = likelihood + prior
    grad = tree_utils.tree_add(likelihood_grad, prior_grad)
    return log_prob, grad, likelihood, net_state

  hmc_update = hmc.make_adaptive_hmc_update(
    _perdevice_log_prob_and_grad, log_prior_diff_fn)

  @functools.partial(
    jax.pmap, axis_name='i', static_broadcasted_argnums=[3, 5, 6, 7, 8],
    in_axes=(0, None, 0, None, None, None, None, None, None)
  )
  def pmap_update(
      dataset, params, net_state, log_likelihood, state_grad, key, step_size,
      trajectory_len, do_mh_correction
  ):
    (params, net_state, log_likelihood, state_grad, step_size, accept_prob,
     accepted) = hmc_update(
        dataset, params, net_state, log_likelihood, state_grad, key, step_size,
        trajectory_len, target_accept_rate=target_accept_rate,
        step_size_adaptation_speed=step_size_adaptation_speed,
        do_mh_correction=do_mh_correction)
    key, = jax.random.split(key, 1)
    return (params, net_state, log_likelihood, state_grad, step_size, key,
            accept_prob, accepted)
  
  def update(
      dataset, params, net_state, log_likelihood, state_grad, key, step_size,
      trajectory_len, do_mh_correction
  ):
    (params, net_state, log_likelihood, state_grad, step_size, key,
     accept_prob, accepted) = pmap_update(
        dataset, params, net_state, log_likelihood, state_grad, key, step_size,
        trajectory_len, do_mh_correction)
    params, state_grad = map(
        tree_utils.get_first_elem_in_sharded_tree, [params, state_grad])
    log_likelihood, step_size, key, accept_prob, accepted = map(
        lambda arr: arr[0],
        [log_likelihood, step_size, key, accept_prob, accepted])
    return (params, net_state, log_likelihood, state_grad, step_size, key,
            accept_prob, accepted)
  
  def get_log_prob_and_grad(params, net_state, dataset):
    pmap_log_prob_and_grad = (
        jax.pmap(
            _perdevice_log_prob_and_grad, axis_name='i', in_axes=(0, None, 0)))
    log_prob, grad, likelihood, net_state = pmap_log_prob_and_grad(
        params, net_state, dataset)
    return (*map(tree_utils.get_first_elem_in_sharded_tree, (log_prob, grad)),
            likelihood[0], net_state)

  return (update, get_log_prob_and_grad,
          make_eval_fn(likelihood_prior_and_acc_fn))


# TODO: rewrite analogously to HMC
def make_sgd_train_epoch(
    net_apply, log_likelihood_fn, log_prior_fn, optimizer, num_batches
):
  """
  Make a training epoch function for SGD-like optimizers.
  """
  perdevice_likelihood_prior_and_grads_fn, likelihood_prior_and_acc_fn = (
    make_perdevice_log_prob_acc_grad_fns(
      net_apply, log_likelihood_fn, log_prior_fn))
  
  def _perdevice_log_prob_and_grad(dataset, params, net_state):
    # Only call inside pmap
    likelihood, likelihood_grad, prior, prior_grad, _, net_state = (
        perdevice_likelihood_prior_and_grads_fn(params, net_state, dataset))
    likelihood = jax.lax.psum(likelihood, axis_name='i')
    likelihood_grad = jax.lax.psum(likelihood_grad, axis_name='i')
    
    log_prob = likelihood * num_batches + prior
    grad = jax.tree_multimap(
      lambda gl, gp: gl * num_batches + gp, likelihood_grad, prior_grad)
    return log_prob, grad, net_state
  
  @functools.partial(
    jax.pmap, axis_name='i', static_broadcasted_argnums=[],
    in_axes=(None, 0, None, 0, 0)
  )
  def pmap_sgd_train_epoch(params, net_state, opt_state, train_set, key):
    n_data = train_set[0].shape[0]
    batch_size = n_data // num_batches
    indices = jax.random.permutation(key, jnp.arange(n_data))
    indices = jax.tree_map(
        lambda x: x.reshape((num_batches, batch_size)), indices)
    
    def train_step(carry, batch_indices):
      batch = jax.tree_map(lambda x: x[batch_indices], train_set)
      params_, net_state_, opt_state_ = carry
      loss, grad, net_state_ = _perdevice_log_prob_and_grad(
        batch, params_, net_state_)
      grad = jax.lax.psum(grad, axis_name='i')
      
      updates, opt_state_ = optimizer.update(grad, opt_state_)
      params_ = optix.apply_updates(params_, updates)
      return (params_, net_state_, opt_state_), loss
    
    (params, net_state, opt_state), losses = jax.lax.scan(
        train_step, (params, net_state, opt_state), indices)
    
    new_key, = jax.random.split(key, 1)
    return losses, params, net_state, opt_state, new_key

  def sgd_train_epoch(params, net_state, opt_state, train_set, key):
    losses, params, net_state, opt_state, new_key = (
      pmap_sgd_train_epoch(params, net_state, opt_state, train_set, key)
    )
    params, opt_state = map(
      tree_utils.get_first_elem_in_sharded_tree, [params, opt_state])
    loss_avg = jnp.mean(losses)
    return params, net_state, opt_state, loss_avg, new_key
  
  return sgd_train_epoch, make_eval_fn(likelihood_prior_and_acc_fn)


@functools.partial(
    jax.pmap, axis_name='i', static_broadcasted_argnums=[0, 4, 5],
    in_axes=(None, None, 0, 0, None, None)
)
def get_softmax_predictions(
    net_apply, params, net_state, dataset, num_batches=1, is_training=False
):
  """Compute predictions for a given network on a given dataset."""

  batch_size = dataset[0].shape[0] // num_batches
  dataset = jax.tree_map(
      lambda x: x.reshape((num_batches, batch_size, *x.shape[1:])), dataset)

  def get_batch_predictions(current_net_state, x):
    y, current_net_state = net_apply(
        params, current_net_state, None, x, is_training)
    batch_predictions = jax.nn.softmax(y)
    return current_net_state, batch_predictions

  _, predictions = jax.lax.scan(get_batch_predictions, net_state, dataset)

  return predictions

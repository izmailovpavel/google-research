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
import tensorflow.compat.v2 as tf
import numpy as onp
import functools
from jax.config import config

from bnn_hmc.core import hmc
from bnn_hmc.core import data
from bnn_hmc.core import losses
from bnn_hmc.utils import tree_utils
from bnn_hmc.utils import ensemble_utils


def set_up_jax(tpu_ip, use_float64):
  if tpu_ip is not None:
    config.FLAGS.jax_xla_backend = "tpu_driver"
    config.FLAGS.jax_backend_target = "grpc://{}:8470".format(tpu_ip)
  if use_float64:
     config.update("jax_enable_x64", True)
  tf.config.set_visible_devices([], 'GPU')


def make_cosine_lr_schedule(init_lr, total_steps):
  """Cosine LR schedule."""
  def schedule(step):
    t = step / total_steps
    return 0.5 * init_lr * (1 + jnp.cos(t * onp.pi))
  return schedule


def make_cosine_lr_schedule_with_burnin(
    init_lr, final_lr, burnin_steps
):
  """Cosine LR schedule with burn-in for SG-MCMC."""
  def schedule(step):
    t = jnp.minimum(step / burnin_steps, 1.)
    coef = (1 + jnp.cos(t * onp.pi)) * 0.5
    return coef * init_lr + (1 - coef) * final_lr
  return schedule


def make_optimizer(lr_schedule, momentum_decay):
  """Make SGD optimizer with momentum."""
  # Maximize log-prob instead of minimizing loss
  return optix.chain(optix.trace(decay=momentum_decay, nesterov=False),
                     optix.scale_by_schedule(lr_schedule))


def get_task_specific_fns(task):
  if task == data.Task.CLASSIFICATION:
    likelihood_fn = losses.make_xent_log_likelihood
    ensemble_fn = ensemble_utils.update_ensemble_classification
  elif task == data.Task.REGRESSION:
    likelihood_fn = losses.make_gaussian_likelihood
    ensemble_fn = ensemble_utils.update_ensemble_regression
  return likelihood_fn, ensemble_fn


def _make_perdevice_likelihood_prior_acc_grad_fns(
    net_apply, log_likelihood_fn, log_prior_fn
):
  """Make functions for training and evaluation.
  
  Functions return likelihood, prior and gradients separately. These values
  can be combined differently for full-batch and mini-batch methods.
  """
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


def _make_perdevice_minibatch_log_prob_and_grad(
    perdevice_likelihood_prior_and_grads_fn, num_batches
):
  """Make log-prob and grad function for mini-batch methods."""
  def perdevice_log_prob_and_grad(dataset, params, net_state):
      likelihood, likelihood_grad, prior, prior_grad, _, net_state = (
        perdevice_likelihood_prior_and_grads_fn(params, net_state, dataset))
      likelihood = jax.lax.psum(likelihood, axis_name='i')
      likelihood_grad = jax.lax.psum(likelihood_grad, axis_name='i')
    
      log_prob = likelihood * num_batches + prior
      grad = jax.tree_multimap(
        lambda gl, gp: gl * num_batches + gp, likelihood_grad, prior_grad)
      return log_prob, grad, net_state
  return perdevice_log_prob_and_grad


def _make_eval_fn(likelihood_prior_and_acc_fn):
  """Define evaluation function."""
  @functools.partial(
    jax.pmap, axis_name='i', in_axes=(None, 0, 0)
  )
  def pmap_eval(params, net_state, dataset):
    likelihood, prior, stats, _ = likelihood_prior_and_acc_fn(
      params, net_state, dataset, is_training=False)
    likelihood = jax.lax.psum(likelihood, axis_name='i')
    log_prob = likelihood + prior
    stats = {
      key: jax.lax.pmean(val, axis_name='i') for key, val in stats.items()}
    stats["likelihood"] = likelihood
    stats["prior"] = prior
    stats["log_prob"] = log_prob
    return stats
  
  def evaluate(params, net_state, dataset):
    stats = pmap_eval(params, net_state, dataset)
    return {key: val[0] for key, val in stats.items()}
  
  return evaluate


def make_hmc_update(
    net_apply, log_likelihood_fn, log_prior_fn, log_prior_diff_fn,
    max_num_leapfrog_steps, target_accept_rate, step_size_adaptation_speed
):
  """Make update and ev0al functions for HMC training."""

  perdevice_likelihood_prior_and_grads_fn, likelihood_prior_and_acc_fn = (
    _make_perdevice_likelihood_prior_acc_grad_fns(
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
      n_leapfrog_steps, do_mh_correction
  ):
    (params, net_state, log_likelihood, state_grad, step_size, accept_prob,
     accepted) = hmc_update(
        dataset, params, net_state, log_likelihood, state_grad, key, step_size,
        n_leapfrog_steps, target_accept_rate=target_accept_rate,
        step_size_adaptation_speed=step_size_adaptation_speed,
        do_mh_correction=do_mh_correction)
    key, = jax.random.split(key, 1)
    return (params, net_state, log_likelihood, state_grad, step_size, key,
            accept_prob, accepted)
  
  def update(
      dataset, params, net_state, log_likelihood, state_grad, key, step_size,
      trajectory_len, do_mh_correction
  ):
    n_leapfrog = jnp.array(jnp.ceil(trajectory_len / step_size), jnp.int32)
    assert n_leapfrog <= max_num_leapfrog_steps, (
      "The trajectory length results in number of leapfrog steps {} which is "
      "higher than max_n_leapfrog {}".format(n_leapfrog, max_num_leapfrog_steps)
    )
    
    (params, net_state, log_likelihood, state_grad, step_size, key,
     accept_prob, accepted) = pmap_update(
        dataset, params, net_state, log_likelihood, state_grad, key, step_size,
        n_leapfrog, do_mh_correction)
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
          _make_eval_fn(likelihood_prior_and_acc_fn))


def make_sgd_train_epoch(
    net_apply, log_likelihood_fn, log_prior_fn, optimizer, num_batches
):
  """
  Make a training epoch function for SGD-like optimizers.
  """
  perdevice_likelihood_prior_and_grads_fn, likelihood_prior_and_acc_fn = (
    _make_perdevice_likelihood_prior_acc_grad_fns(
      net_apply, log_likelihood_fn, log_prior_fn))

  _perdevice_log_prob_and_grad = _make_perdevice_minibatch_log_prob_and_grad(
      perdevice_likelihood_prior_and_grads_fn, num_batches
  )
  
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
  
  return sgd_train_epoch, _make_eval_fn(likelihood_prior_and_acc_fn)


def make_get_predictions(activation_fn):
  @functools.partial(
    jax.pmap, axis_name='i', static_broadcasted_argnums=[0, 4, 5],
    in_axes=(None, None, 0, 0, None, None)
  )
  def get_predictions(
      net_apply, params, net_state, dataset, num_batches=1, is_training=False
  ):
    batch_size = dataset[0].shape[0] // num_batches
    dataset = jax.tree_map(
      lambda x: x.reshape((num_batches, batch_size, *x.shape[1:])), dataset)

    def get_batch_predictions(current_net_state, x):
      y, current_net_state = net_apply(
        params, current_net_state, None, x, is_training)
      batch_predictions = activation_fn(y)
      return current_net_state, batch_predictions

    _, predictions = jax.lax.scan(get_batch_predictions, net_state, dataset)
    predictions = predictions.reshape((
        num_batches * batch_size, *predictions.shape[2:]))
    return predictions
  return get_predictions


get_softmax_predictions = make_get_predictions(jax.nn.softmax)
get_regression_gaussian_predictions = make_get_predictions(
    losses.preprocess_network_outputs_gaussian)

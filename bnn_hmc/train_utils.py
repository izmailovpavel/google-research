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
import pickle
import re

from bnn_hmc import hmc
from bnn_hmc import nn_loss


LRSchedule = Callable[[jnp.ndarray], jnp.ndarray]
Opt = optix.GradientTransformation
_CHECKPOINT_FORMAT_STRING = "model_step_{}.pt"


def make_cosine_lr_schedule(init_lr,
                            total_steps):
  """Cosine LR schedule."""
  def schedule(step):
    t = step / total_steps
    return 0.5 * init_lr * (1 + jnp.cos(t * onp.pi))
  return schedule


def make_optimizer(lr_schedule, momentum_decay):
  return optix.chain(optix.trace(decay=momentum_decay, nesterov=False),
                     optix.scale_by_schedule(lr_schedule),
                     optix.scale(-1))


def make_hmc_update_eval_fns(
    net,
    train_set,
    test_set,
    log_likelihood_fn,
    log_prior_fn,
    log_prior_diff_fn,
    target_accept_rate,
    step_size_adaptation_speed
):
  """Make update and ev0al functions for HMC training."""

  def log_prob_and_grad_fn(params):

    likelihood, likelihood_grad = nn_loss.pmap_get_log_likelihood_and_grad(
        net, params, log_likelihood_fn, train_set)
    prior, prior_grad = jax.value_and_grad(log_prior_fn)(params)
    log_prob = likelihood[0] + prior
    grad = jax.tree_multimap(lambda g_l, g_p: g_l[0] + g_p,
                             likelihood_grad, prior_grad)
    return log_prob, grad, likelihood[0], prior

  def log_prob_and_acc(params, dataset):
    log_prob, acc = nn_loss.pmap_get_log_prob_and_accuracy(
        net, params, log_likelihood_fn, log_prior_fn, dataset)
    return log_prob[0], acc[0]

  hmc_update = hmc.make_adaptive_hmc_update(
      log_prob_and_grad_fn, log_prior_diff_fn)

  def update(
      params, log_likelihood, state_grad, key, step_size, trajectory_len,
      do_mh_correction):
    params, log_likelihood, state_grad, step_size, accept_prob = hmc_update(
        params, log_likelihood, state_grad, key, step_size, trajectory_len,
        target_accept_rate=target_accept_rate,
        step_size_adaptation_speed=step_size_adaptation_speed,
        do_mh_correction=do_mh_correction)
    key, = jax.random.split(key, 1)
    return params, log_likelihood, state_grad, step_size, key, accept_prob

  def evaluate(params):
    test_log_prob, test_acc = log_prob_and_acc(params, test_set)
    train_log_prob, train_acc = log_prob_and_acc(params, train_set)
    return test_log_prob, test_acc, train_log_prob, train_acc

  return update, evaluate, log_prob_and_grad_fn


def make_checkpoint_dict(params, key, step_size, trajectory_len):
  checkpoint_dict = {
      "params": params,
      "key": key,
      "step_size": step_size,
      "traj_len": trajectory_len
  }
  return checkpoint_dict


def parse_checkpoint_dict(checkpoint_dict):
  field_names = ["params", "key", "step_size", "traj_len"]
  return [checkpoint_dict[name] for name in field_names]


def load_checkpoint(path):
    with open(path, "rb") as f:
        checkpoint_dict = pickle.load(f)
    return parse_checkpoint_dict(checkpoint_dict)


def save_checkpoint(path, checkpoint_dict):
    with open(path, "wb") as f:
        pickle.dump(checkpoint_dict, f)


def _checkpoint_pattern():
    pattern_string = _CHECKPOINT_FORMAT_STRING.format("(?P<step>[0-9]+)")
    return re.compile(pattern_string)


def _match_checkpoint_pattern(name):
    pattern = _checkpoint_pattern()
    return pattern.match(name)


def name_is_checkpoint(name):
    return bool(_match_checkpoint_pattern(name))


def parse_checkpoint_name(name):
    match = _match_checkpoint_pattern(name)
    return int(match.group("step"))


def make_checkpoint_name(step):
    return _CHECKPOINT_FORMAT_STRING.format(step)

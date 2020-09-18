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

"""Implementation of Hamiltonian Monte Carlo."""


import jax
import jax.numpy as jnp


def make_leapfrog(log_prob_and_grad):
  """Leapfrog method."""

  # Return likelihood and prior separately in log_prob_and_grad to compute the
  # prior densities ratio more accurately in float32 in the accept-reject step.
  def _leapfrog_body(step_size, state, momentum, state_grad):
    momentum = jax.tree_multimap(lambda m, g: m + 0.5 * step_size * g, momentum,
                                 state_grad)
    state = jax.tree_multimap(lambda s, m: s + m * step_size, state, momentum)
    state_log_prob, state_grad, log_likelihood, _ = log_prob_and_grad(state)
    momentum = jax.tree_multimap(lambda m, g: m + 0.5 * step_size * g, momentum,
                                 state_grad)

    return state, momentum, state_grad, state_log_prob, log_likelihood

  def leapfrog(step_size, n_leapfrog, state, momentum, state_grad):
    # Do not use `lax.fori_loop` to avoid jit-of-pmap.
    for _ in range(n_leapfrog):
      state, momentum, state_grad, state_log_prob, state_likelihood = \
          _leapfrog_body(step_size, state, momentum, state_grad)
    return state, momentum, state_grad, state_likelihood

  return leapfrog


def _nan_to_inf(x):
  return jnp.where(jnp.isnan(x), jnp.inf + jnp.zeros_like(x), x)


def _first(xy):
  return xy[0]


def _second(xy):
  return xy[1]


def get_kinetic_energy_diff(momentum1, momentum2):
  return sum([0.5 * jnp.sum(m1**2 - m2**2) for m1, m2 in
              zip(jax.tree_leaves(momentum1), jax.tree_leaves(momentum2))])

def make_accept_prob(log_prior_diff_fn):
  def get_accept_prob(
      log_likelihood1, state1, momentum1, log_likelihood2, state2, momentum2
    ):
    energy_diff = get_kinetic_energy_diff(momentum1, momentum2)
    energy_diff -= log_likelihood1 - log_likelihood2
    energy_diff -= log_prior_diff_fn(state1, new_state2)
    accept_prob = jnp.minimum(1., jnp.exp(energy_diff))
    return accept_prob


def make_adaptive_hmc_update(log_prob_and_grad_fn, log_prior_diff_fn):
  """Returns an adaptive HMC update function."""
  leapfrog = make_leapfrog(log_prob_and_grad_fn)
  get_accept_prob = make_accept_prob(log_prior_diff_fn)

  def get_kinetic_energy(momentum):
    return sum([0.5 * jnp.sum(m**2) for m in jax.tree_leaves(momentum)])


  def adaptive_hmc_update(
      state,
      log_likelihood,
      state_grad,
      key,
      step_size,
      trajectory_len,
      target_accept_rate=0.5,
      step_size_adaptation_speed=0.05,
      max_n_leapfrog=1000,
      jitter_amt=0.2
  ):

    normal_key, uniform_key, jitter_key = jax.random.split(key, 3)

    n_leapfrog = jnp.array(jnp.ceil(trajectory_len / step_size), jnp.int32)
    n_leapfrog = jnp.minimum(n_leapfrog, max_n_leapfrog)
    jittered_step_size = step_size * jnp.exp(
        jnp.where(
            jnp.logical_or(step_size_adaptation_speed <= 0,
                           target_accept_rate <= 0),
            jnp.log(1. + jitter_amt) * (2 * jax.random.uniform(jitter_key,
                                                               ()) - 1.), 0.))

    num_leaves = len(jax.tree_leaves(state))
    normal_keys = list(jax.random.split(normal_key, num_leaves))
    treedef = jax.tree_structure(state)
    normal_keys = jax.tree_unflatten(treedef, normal_keys)
    momentum = jax.tree_multimap(lambda s, key: jax.random.normal(key, s.shape),
                                 state, normal_keys)

    new_state, new_momentum, new_grad, new_log_likelihood = leapfrog(
        jittered_step_size, n_leapfrog, state, momentum, state_grad)

    #log_prob = log_prob_and_grad_fn(state)[0]
    #new_log_prob = log_prob_and_grad_fn(new_state)[0]
    #initial_energy = get_kinetic_energy(momentum) - log_prob
    #new_energy = _nan_to_inf(get_kinetic_energy(new_momentum) - new_log_prob)
    #energy_diff_old = initial_energy - new_energy
    #accept_prob_old = jnp.minimum(1., jnp.exp(energy_diff_old))

    #energy_diff = get_kinetic_energy_diff(momentum, new_momentum)
    #energy_diff -= log_likelihood - new_log_likelihood
    #energy_diff -= log_prior_diff_fn(state, new_state)

    #accept_prob = jnp.minimum(1., jnp.exp(energy_diff))

    #print("Accept prob", accept_prob, accept_prob_old)
    #print("Energy diff", energy_diff, energy_diff_old)
    #print("Kinetic energy diff", get_kinetic_energy_diff(momentum, new_momentum), get_kinetic_energy(momentum) - get_kinetic_energy(new_momentum))
    #print("Prior diff", log_prior_diff_fn(state, new_state))
    #print("Likelihood diff", log_likelihood - new_log_likelihood)
    #print("Logprob diff", log_prob - new_log_prob)
    # TODO(izmailovpavel): check why the second condition is needed.
    accepted = jnp.logical_and(
        jax.random.uniform(uniform_key, log_prob.shape) < accept_prob,
        jnp.isfinite(energy_diff))

    step_size = step_size * jnp.exp(
        jnp.where(
            jnp.logical_or(target_accept_rate <= 0,
                           step_size_adaptation_speed <= 0), 0.,
            step_size_adaptation_speed *
            (jnp.mean(accept_prob) - target_accept_rate)))

    state = jax.lax.cond(accepted, _first, _second, (new_state, state))
    log_likelihood = jnp.where(accepted, new_log_likelihood, log_likelihood)
    state_grad = jax.lax.cond(accepted, _first, _second, (new_grad, state_grad))
    return state, log_likelihood, state_grad, step_size, accept_prob
  return adaptive_hmc_update

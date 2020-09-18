import unittest
from jax.config import config
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp
import argparse
from jax.experimental.callback import rewrite

from bnn_hmc import data
from bnn_hmc import models
from bnn_hmc import nn_loss
from bnn_hmc import precision_utils
from bnn_hmc import train_utils
from bnn_hmc import hmc

parser = argparse.ArgumentParser("Unit tests for BNN HMC code.")
parser.add_argument("--tpu_ip", type=str, default="10.0.0.2",
                    help="Cloud TPU internal ip "
                         "(see `gcloud compute tpus list`)")

args = parser.parse_args()
config.FLAGS.jax_xla_backend = "tpu_driver"
config.FLAGS.jax_backend_target = "grpc://{}:8470".format(args.tpu_ip)



class TestHMC(unittest.TestCase):
  
  @staticmethod
  def _prepare():
    net_fn = models.lenet_fn
    net = hk.transform(net_fn)
    train_set, test_set, _ = data.make_ds_pmap_fullbatch(name="cifar10")
    init_key = jax.random.PRNGKey(0)
    init_data = jax.tree_map(lambda elem: elem[0], train_set)
    params = net.init(init_key, init_data)
    return net, params, train_set, test_set, init_data, init_key

  def test_rewrite_callback_changes_predictions(self):
    """Test that the rewrite callback is changing predictions."""
    net, params, _, test_set, _, _ = self._prepare()
    net_fn_highprec = jax.experimental.callback.rewrite(
        models.lenet_fn,
        precision_utils.HIGH_PRECISION_RULES)
    net_highprec = hk.transform(net_fn_highprec)
    predictions = nn_loss.pmap_get_softmax_predictions(net, params, test_set, 1)
    predictions_highprec = nn_loss.pmap_get_softmax_predictions(
        net_highprec, params, test_set, 1)
    predictions_dist = jnp.sqrt(
        jnp.sum((predictions - predictions_highprec)**2))
    self.assertGreater(predictions_dist, 1e-2)

  def test_no_randomness_in_predictions(self):
    """Test that computing predictions twice gives the same results."""
    net, params, _, test_set, _, _ = self._prepare()
    predictions = nn_loss.pmap_get_softmax_predictions(net, params, test_set, 1)
    predictions2 = nn_loss.pmap_get_softmax_predictions(net, params, test_set, 1)
    predictions_dist = jnp.sqrt(jnp.sum((predictions - predictions2) ** 2))
    self.assertLess(predictions_dist, 1e-6)
  
  def test_gaussian_prior_difference_precision(self):
    """Test that we compute gaussian prior difference with high precision"""
    weight_decay = 30.
    net, params, _, _, init_data, init_key = self._prepare()
    init_key, = jax.random.split(init_key, 1)
    params2 = net.init(init_key, init_data)
    _, prior_diff_fn = nn_loss.make_gaussian_log_prior(
        weight_decay=weight_decay)

    def prior_diff_onp_fn(params1, params2):
      diff = sum([onp.sum(onp.array(p1)**2 - onp.array(p2)**2) for p1, p2 in
                  zip(jax.tree_leaves(params1), jax.tree_leaves(params2))])
      return -0.5 * weight_decay * diff

    prior_diff = prior_diff_fn(params, params2)
    prior_diff_np = prior_diff_onp_fn(params, params2)
    self.assertLess(jnp.abs(prior_diff_np - prior_diff), 1e-2)
  
  def test_log_prob_fn(self):
      """Test log-prob fn constructed by train_utils."""
      weight_decay = 30.
      net, params, train_set, test_set, _, _ = self._prepare()
      log_likelihood_fn = nn_loss.xent_log_likelihood
      log_prior_fn, log_prior_diff = (
          nn_loss.make_gaussian_log_prior(weight_decay=weight_decay))

      def get_log_prob(dataset):
        _, _, fn = (
            train_utils.make_hmc_update_eval_fns(
                net, dataset, test_set, log_likelihood_fn, log_prior_fn,
                log_prior_diff))
        return fn

      log_prob_and_grad_fn = get_log_prob(train_set)
      log_prob, grad, likelihood, prior = log_prob_and_grad_fn(params)
      self.assertEqual(log_prob, likelihood + prior)

      n_split = train_set[0].shape[1]
      first_half = jax.tree_map(lambda x: x[:, :n_split], train_set)
      second_half = jax.tree_map(lambda x: x[:, n_split:], train_set)
      log_prob_and_grad_first_half_fn = get_log_prob(first_half)
      log_prob_and_grad_second_half_fn = get_log_prob(second_half)
      log_prob_fh, grad_fh, likelihood_fh, prior_fh = (
          log_prob_and_grad_first_half_fn(params))

      log_prob_sh, grad_sh, likelihood_sh, prior_sh = (
        log_prob_and_grad_second_half_fn(params))
    
      self.assertEqual(likelihood, likelihood_fh+likelihood_sh)
      self.assertEqual(prior, prior_fh)
      self.assertEqual(prior, prior_sh)
      self.assertEqual(log_prob, log_prob_fh + log_prob_sh - prior_sh)
      
  def test_accept_prob(self):
    """Test make_accept_prob implementation in hmc."""
    weight_decay = 30.
    net, params, train_set, test_set, init_data, init_key = self._prepare()
    init_key, = jax.random.split(init_key, 1)
    params2 = net.init(init_key, init_data)

    # Rescale parameters so accept_prob is not 0 or 1.
    params, params2 = jax.tree_map(lambda p: p * 1e-2, [params, params2])

    log_likelihood_fn = nn_loss.xent_log_likelihood
    log_prior_fn, log_prior_diff = (
        nn_loss.make_gaussian_log_prior(weight_decay=weight_decay))
    _, _, log_prob_and_grad_fn = (
      train_utils.make_hmc_update_eval_fns(
        net, train_set, test_set, log_likelihood_fn, log_prior_fn,
        log_prior_diff))

    log_prob, _, log_likelihood, log_prior = log_prob_and_grad_fn(params)
    log_prob2, _, log_likelihood2, log_prior2 = log_prob_and_grad_fn(params2)
    
    key, key2 = jax.random.split(init_key, 2)
    momentum = hmc.sample_momentum(params, key)
    momentum2 = hmc.sample_momentum(params2, key2)
    momentum, momentum2 = jax.tree_map(lambda p: p * 1e-2, [momentum, momentum2])

    get_accept_prob = hmc.make_accept_prob(log_prior_diff)
    accept_prob = get_accept_prob(
        log_likelihood, params, momentum, log_likelihood2, params2, momentum2)
    accept_prob_reverse = get_accept_prob(
      log_likelihood2, params2, momentum2, log_likelihood, params, momentum)
    
    def prior_onp_fn(params):
      norm_sq = sum([onp.sum(onp.array(p, dtype=onp.float128)**2)
                     for p in jax.tree_leaves(params)])
      return -0.5 * weight_decay * norm_sq
    
    def kinetic_energy_onp_fn(momentum):
      return sum([0.5 * onp.sum(onp.array(m, dtype=onp.float128)**2)
                  for m in jax.tree_leaves(momentum)])
    
    energy = (
        kinetic_energy_onp_fn(momentum) -
        onp.array(log_likelihood, dtype=onp.float128) - prior_onp_fn(params))
    energy2 = (
        kinetic_energy_onp_fn(momentum2) -
        onp.array(log_likelihood2, dtype=onp.float128) - prior_onp_fn(params2))
    energy_diff = energy - energy2
    onp_accept_prob = onp.minimum(1., onp.exp(energy_diff))
    onp_accept_prob_reverse = onp.minimum(1., onp.exp(-energy_diff))

    self.assertLess(
      onp.abs(onp_accept_prob - 
              onp.array(accept_prob, dtype=onp.float128)), 1e-4)
    self.assertLess(
      onp.abs(onp_accept_prob_reverse - 
              onp.array(accept_prob_reverse, dtype=onp.float128)), 1e-4)

    accept_prob_not_1_or_0 = (
        ((accept_prob > 1e-4) and (accept_prob < 1. - 1e-4)) or
        ((accept_prob_reverse > 1e-4) and (accept_prob_reverse < 1. - 1e-4)))
    assert accept_prob_not_1_or_0


if __name__ == '__main__':
  unittest.main()

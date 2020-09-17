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

parser = argparse.ArgumentParser("Unit tests for BNN HMC code.")
parser.add_argument("--tpu_ip", type=str, default="10.0.0.2",
                    help="Cloud TPU internal ip "
                         "(see `gcloud compute tpus list`)")

args = parser.parse_args()
config.FLAGS.jax_xla_backend = "tpu_driver"
config.FLAGS.jax_backend_target = "grpc://{}:8470".format(args.tpu_ip)


class TestPrecision(unittest.TestCase):

  @staticmethod
  def _prepare(self):
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
    predictions = nn_loss.pmap_get_softmax_preds(net, params, test_set, 1)
    predictions_highprec = nn_loss.pmap_get_softmax_preds(
        net_highprec, params, test_set, 1)
    predictions_dist = jnp.sqrt(
        jnp.sum((predictions - predictions_highprec)**2))
    self.assertGreater(predictions_dist, 1e-2)

  def test_no_randomness_in_predictions(self):
    """Test that computing predicitons twice gives the same results."""
    net, params, _, test_set, _, _ = self._prepare()
    predictions = nn_loss.pmap_get_softmax_preds(net, params, test_set, 1)
    predictions2 = nn_loss.pmap_get_softmax_preds(net, params, test_set, 1)
    predictions_dist = jnp.sqrt(jnp.sum((predictions - predictions2) ** 2))
    self.assertLess(predictions_dist, 1e-6)
  
  def test_gaussian_prior_difference_precision(self):
    """Test that we compute gaussian prior difference with high precision"""
    net, params, _, _, init_data, init_key = self._prepare()
    init_key = jax.random.split(init_key, 1)
    params2 = net.init(init_key, init_data)
    weight_decay = 30.
    prior_diff_fn = nn_loss.make_gaussian_prior_difference(
        weight_decay=weight_decay)

    def prior_diff_onp_fn(params1, params2):
      """Computes the delta in  Gaussian prior negative log-density."""
      diff = sum([onp.sum(onp.array(p1) ** 2 - onp.array(p2) ** 2) for p1, p2 in
                  zip(jax.tree_leaves(params1), jax.tree_leaves(params2))])
      return 0.5 * weight_decay * diff

    prior_diff = prior_diff_fn(params, params2)
    prior_diff_np = prior_diff_onp_fn(params, params2)
    self.assertLess(jnp.abs(prior_diff_np - prior_diff), 1e-2)


if __name__ == '__main__':
  unittest.main()

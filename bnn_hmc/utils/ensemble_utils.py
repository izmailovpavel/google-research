from bnn_hmc.utils import metrics
from bnn_hmc.utils import train_utils
import numpy as onp


def running_average(old_avg_val, new_val, n_avg):
  new_avg_val = old_avg_val + (new_val - old_avg_val) / (n_avg + 1)
  return new_avg_val


def compute_updated_ensemble_predictions_classification(
    ensemble_predicted_probs, num_ensembled, new_predicted_probs
):
  """Update ensemble predictive categorical distribution."""
  #ToDo: test
  if num_ensembled:
    new_ensemble_predicted_probs = running_average(
        ensemble_predicted_probs, new_predicted_probs, num_ensembled)
  else:
    new_ensemble_predicted_probs = new_predicted_probs
  return new_ensemble_predicted_probs


def update_ensemble_classification(
    net_apply, params, net_state, test_set, num_ensembled,
    ensemble_predicted_probs
):
  """Update ensemble predicted probabilities for classification."""
  predicted_probs = onp.asarray(train_utils.get_softmax_predictions(
      net_apply, params, net_state, test_set, 1, False))

  new_ensemble_predicted_probs = (
      compute_updated_ensemble_predictions_classification(
        ensemble_predicted_probs, num_ensembled, predicted_probs))
  test_labels = test_set[1]

  stats = {
    "accuracy": metrics.accuracy(new_ensemble_predicted_probs, test_labels),
    "nll": metrics.nll(new_ensemble_predicted_probs, test_labels),
    "ece": metrics.calibration_curve(new_ensemble_predicted_probs,
                                     test_labels)["ece"],
    "num_samples": num_ensembled + 1
  }

  return new_ensemble_predicted_probs, num_ensembled + 1, stats


def compute_updated_ensemble_predictions_regression(
    ensemble_predictions, num_ensembled, new_predictions
):
  """Update ensemble predictive distribution assuming Gaussian likelihood."""
  mus, sigmas = onp.split(new_predictions, [1], axis=-1)

  if num_ensembled:
    old_mus, old_sigmas = onp.split(ensemble_predictions, [1], axis=-1)
    new_mus = running_average(old_mus, mus, num_ensembled)
    old_sigmas_corrected = old_sigmas + old_mus ** 2 - new_mus ** 2
    new_sigmas = running_average(
      old_sigmas_corrected, sigmas + mus ** 2 - new_mus ** 2, num_ensembled)
    new_ensemble_predictions = onp.concatenate([new_mus, new_sigmas], axis=-1)
  else:
    new_ensemble_predictions = new_predictions
  return new_ensemble_predictions


def update_ensemble_regression(
    net_apply, params, net_state, test_set, num_ensembled,
    ensemble_predictions
):
  """Update ensemble predicted probabilities for regression."""
  # ToDo: Test!
  new_predictions = onp.asarray(train_utils.get_regression_gaussian_predictions(
      net_apply, params, net_state, test_set, 1, False))

  new_ensemble_predictions = (
      compute_updated_ensemble_predictions_regression(
        ensemble_predictions, num_ensembled, new_predictions))
  test_targets = test_set[1]

  stats = {
    "mse_of_mean": metrics.mse(new_ensemble_predictions, test_targets),
    "mse_of_mean": metrics.mse(new_ensemble_predictions, test_targets),
    "nll": metrics.regression_nlls(new_ensemble_predictions, test_targets),
    "num_samples": num_ensembled + 1
  }

  return new_ensemble_predictions, num_ensembled + 1, stats

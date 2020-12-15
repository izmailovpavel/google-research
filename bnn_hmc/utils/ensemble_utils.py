from bnn_hmc.utils import metrics
from bnn_hmc.utils import train_utils
import numpy as onp


def running_average(old_avg_val, new_val, n_avg):
  new_avg_val = old_avg_val + (new_val - old_avg_val) / (n_avg + 1)
  return new_avg_val


def update_ensemble_classification(
    net_apply, params, net_state, test_set, num_ensembled,
    ensemble_predicted_probs
):
  """Update ensemble predicted probabilities for classification."""
  predicted_probs = onp.asarray(train_utils.get_softmax_predictions(
      net_apply, params, net_state, test_set, 1, False))
  if num_ensembled:
    new_ensemble_predicted_probs = running_average(
        ensemble_predicted_probs, predicted_probs, num_ensembled)
  else:
    new_ensemble_predicted_probs = predicted_probs
  test_labels = test_set[1]

  stats = {
    "accuracy": metrics.accuracy(new_ensemble_predicted_probs, test_labels),
    "nll": metrics.nll(new_ensemble_predicted_probs, test_labels),
    "ece": metrics.calibration_curve(new_ensemble_predicted_probs,
                                     test_labels)["ece"],
    "num_samples": num_ensembled + 1
  }

  return new_ensemble_predicted_probs, num_ensembled + 1, stats


def update_ensemble_regression(
    net_apply, params, net_state, test_set, num_ensembled,
    ensemble_predictions
):
  """Update ensemble predicted probabilities for regression."""
  # ToDo: Test!
  predictions = onp.asarray(train_utils.get_softmax_predictions(
      net_apply, params, net_state, test_set, 1, False))
  mus, sigmas = predictions[:, 0], predictions[:, 1]

  if num_ensembled:
    old_mus, old_sigmas = ensemble_predictions[:, 0], ensemble_predictions[:, 1]
    new_mus = running_average(old_mus, mus, num_ensembled)
    old_sigmas_corrected = old_sigmas + old_mus**2 - new_mus**2
    new_sigmas = running_average(
        old_sigmas_corrected, sigmas + mus**2 - new_mus**2, num_ensembled)
    new_ensemble_predictions = onp.hstack(new_mus, new_sigmas)
  else:
    new_ensemble_predictions = predictions
  test_targets = test_set[1]

  stats = {
    "mean_mse": metrics.mse(new_ensemble_predictions[:, 0], test_targets),
    "nll": metrics.nll(
      new_ensemble_predictions[:, 0], new_ensemble_predictions[:, 1],
      test_targets),
    "num_samples": num_ensembled + 1
  }

  return new_ensemble_predictions, num_ensembled + 1, stats

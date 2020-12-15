"""numpy implementations of prediction quality metrics.

Partly adapted from https://github.com/wjmaddox/swa_gaussian/blob/master/experiments/uncertainty/uncertainty.py
"""

import numpy as onp


def _flatten_batch_axes(arr):
  pred_dim = arr.shape[-1]
  batch_dim = arr.size // pred_dim
  return arr.reshape((batch_dim, pred_dim))


def _flatten_outputs_labels(outputs, labels):
  return _flatten_batch_axes(outputs), labels.reshape(-1)


def accuracy(outputs, labels):
  """Negative log-likelihood."""
  outputs, labels = _flatten_outputs_labels(outputs, labels)
  labels = labels.astype(int)
  preds = onp.argmax(outputs, axis=1)
  return (preds == labels).mean()


def nll(outputs, labels, normalized=True):
  """Negative log-likelihood."""
  outputs, labels = _flatten_outputs_labels(outputs, labels)
  labels = labels.astype(int)
  idx = (onp.arange(labels.size), labels)
  log_ps = onp.log(outputs[idx])
  if normalized:
    return -log_ps.mean()
  else:
    return -log_ps.sum()


def calibration_curve(outputs, labels, num_bins=20):
  """Compute calibration curve and ECE."""
  outputs, labels = _flatten_outputs_labels(outputs, labels)
  confidences = onp.max(outputs, 1)
  num_inputs = confidences.shape[0]
  step = (num_inputs + num_bins - 1) // num_bins
  bins = onp.sort(confidences)[::step]
  if num_inputs % step != 1:
    bins = onp.concatenate((bins, [onp.max(confidences)]))
  predictions = onp.argmax(outputs, 1)
  bin_lowers = bins[:-1]
  bin_uppers = bins[1:]

  accuracies = (predictions == labels)

  bin_confidences = []
  bin_accuracies = []
  bin_proportions = []

  ece = 0.0
  for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
    in_bin = (confidences > bin_lower) * (confidences < bin_upper)
    prop_in_bin = in_bin.mean()
    if prop_in_bin > 0:
      accuracy_in_bin = accuracies[in_bin].mean()
      avg_confidence_in_bin = confidences[in_bin].mean()
      ece += onp.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
      bin_confidences.append(avg_confidence_in_bin)
      bin_accuracies.append(accuracy_in_bin)
      bin_proportions.append(prop_in_bin)

  bin_confidences, bin_accuracies, bin_proportions = map(
      lambda lst: onp.array(lst),
      (bin_confidences, bin_accuracies, bin_proportions))
  
  return {
      "confidence": bin_confidences,
      "accuracy": bin_accuracies,
      "proportions": bin_proportions,
      "ece": ece}


def mse(predictions, targets):
  assert predictions.shape == targets.shape, (
    "Predictions and targets should have the same shape, "
    "got {} and {}".format(predictions.shape, targets.shape))
  return ((predictions - targets)**2).mean()


def regression_nlls(prediction_means, predictions_vars, targets):
  #ToDo: check
  se = (prediction_means - targets) ** 2
  nll = -onp.mean(se / (2 * predictions_vars))
  nll -= onp.mean(onp.log(2 * onp.pi * predictions_vars))
  return nll

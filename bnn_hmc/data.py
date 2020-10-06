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

"""Data loaders."""

from typing import Generator, Tuple

import jax
import numpy as onp
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from enum import Enum

# SupervisedDataset = Tuple[onp.ndarray, onp.ndarray]
# SupervisedDatasetGen = Generator[SupervisedDataset, None, None]

_CHECKPOINT_FORMAT_STRING = "model_step_{}.pt"


class ImgDatasets(Enum):
  CIFAR10 = "cifar10"


# Format: (img_mean, img_std)
_ALL_IMG_DS_STATS = {
    ImgDatasets.CIFAR10: ((0.49, 0.48, 0.44), (0.2, 0.2, 0.2))
}

_IMDB_CONFIG = {
  "max_features": 20000,
  "max_len": 100,
  "num_train": 20000
}


def load_imdb_dataset():
  """
  Load the IMDB reviews dataset.
  
  Code adapted from the code for
  _How Good is the Bayes Posterior in Deep Neural Networks Really?_:
  https://github.com/google-research/google-research/blob/master/cold_posterior_bnn/imdb/imdb_data.py
  """
  (x_train, y_train), (x_test, y_test) = imdb.load_data(
      path="./datasets", num_words=_IMDB_CONFIG["max_features"])
  num_train = _IMDB_CONFIG["num_train"]
  x_train, x_val = x_train[:num_train], x_train[num_train:]
  y_train, y_val = y_train[:num_train], y_train[num_train:]

  def preprocess(x, y, max_length):
    x = sequence.pad_sequences(x, maxlen=max_length)
    y = onp.array(y)
    x = onp.array(x)
    # x = tf.convert_to_tensor(x, dtype=tf.int32)
    # y = tf.convert_to_tensor(y, dtype=tf.int32)
    return x, y

  max_length = _IMDB_CONFIG["max_len"]
  x_train, y_train = preprocess(x_train, y_train, max_length=max_length)
  x_val, y_val = preprocess(x_val, y_val, max_length=max_length)
  x_test, y_test = preprocess(x_test, y_test, max_length=max_length)
  return (x_train, y_train), (x_test, y_test), (x_val, y_val), 2


def load_image_dataset(
    split, batch_size, name="cifar10", repeat=False, shuffle=False,
    shuffle_seed=None
):
  """Loads the dataset as a generator of batches."""
  # Do no data augmentation.
  ds, dataset_info = tfds.load(name, split=split, as_supervised=True,
                               with_info=True)
  num_classes = dataset_info.features["label"].num_classes
  num_examples = dataset_info.splits[split].num_examples

  def img_to_float32(image, label):
    return tf.image.convert_image_dtype(image, tf.float32), label

  ds = ds.map(img_to_float32).cache()
  ds_stats = _ALL_IMG_DS_STATS[ImgDatasets(name)]

  def img_normalize(image, label):
    """Normalize the image to zero mean and unit variance."""
    mean, std = ds_stats
    image -= tf.constant(mean, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(std, shape=[1, 1, 3], dtype=image.dtype)
    return image, label

  ds = ds.map(img_normalize)
  if batch_size == -1:
    batch_size = num_examples
  if repeat:
    ds = ds.repeat()
  if shuffle:
    ds = ds.shuffle(buffer_size=10 * batch_size, seed=shuffle_seed)
  ds = ds.batch(batch_size)
  return tfds.as_numpy(ds), num_classes, num_examples


def get_image_dataset(name):
  train_set, n_classes, _ = load_image_dataset("train", -1, name)
  train_set = next(iter(train_set))
  
  test_set, _, _ = load_image_dataset("test", -1, name)
  test_set = next(iter(test_set))
  return train_set, test_set, None, n_classes


def batch_split_axis(batch, n_split):
  """Reshapes batch to have first axes size equal n_split."""
  x, y = batch
  n = x.shape[0]
  n_new = n / n_split
  assert n_new == int(n_new), (
      "First axis cannot be split: batch dimension was {} when "
      "n_split was {}.".format(x.shape[0], n_split))
  n_new = int(n_new)
  return tuple(arr.reshape([n_split, n_new, *arr.shape[1:]]) for arr in (x, y))


def pmap_dataset(ds, n_devices=None):
  """Shard the dataset to devices."""
  n_devices = n_devices or len(jax.local_devices())
  return jax.pmap(lambda x: x)(batch_split_axis(ds, n_devices))
  

def make_ds_pmap_fullbatch(name="cifar10", n_devices=None):
  """Make train and test sets sharded over batch dim."""
  name = name.lower()
  if name in ImgDatasets._value2member_map_:
    train_set, test_set, _, n_classes = get_image_dataset(name)
  elif name == "imdb":
    train_set, test_set, _, n_classes = load_imdb_dataset()
  else:
    raise ValueError("Unknown dataset name: {}".format(name))
  
  train_set, test_set = tuple(pmap_dataset(ds, n_devices)
                              for ds in (train_set, test_set))
  return train_set, test_set, n_classes

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

"""CNN haiku models."""

from typing import Tuple
import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental.callback import rewrite

from bnn_hmc import precision_utils


Batch = Tuple[jnp.ndarray, jnp.ndarray]
_DEFAULT_BN_CONFIG = {
  'decay_rate': 0.9,
  'eps': 1e-5,
  'create_scale': True,
  'create_offset': True
}


def make_lenet_fn(num_classes):
  def lenet_fn(batch, is_training):
    """Network inspired by LeNet-5."""
    x, _ = batch
    x = x.astype(jnp.float32)
  
    cnn = hk.Sequential([
        hk.Conv2D(output_channels=32, kernel_shape=5, padding="SAME"),
        jax.nn.relu,
        hk.MaxPool(window_shape=3, strides=2, padding="VALID"),
        hk.Conv2D(output_channels=64, kernel_shape=5, padding="SAME"),
        jax.nn.relu,
        hk.MaxPool(window_shape=3, strides=2, padding="VALID"),
        hk.Conv2D(output_channels=128, kernel_shape=5, padding="SAME"),
        hk.MaxPool(window_shape=3, strides=2, padding="VALID"),
        hk.Flatten(),
        hk.Linear(1000),
        jax.nn.relu,
        hk.Linear(1000),
        jax.nn.relu,
        hk.Linear(num_classes),
    ])
    return cnn(x)
  return lenet_fn


he_normal = hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal')


def _resnet_layer(
    inputs, num_filters, kernel_size=3, strides=1, activation=lambda x: x,
    use_bias=True, is_training=True, bn_config=_DEFAULT_BN_CONFIG
):
  x = inputs
  x = hk.Conv2D(
      num_filters,
      kernel_size,
      stride=strides,
      padding='same',
      w_init=he_normal,
      with_bias=use_bias)(
          x)
  x = hk.BatchNorm(**bn_config)(x, is_training=is_training)
  x = activation(x)
  return x


def make_resnet_fn(
    num_classes: int,
    depth: int,
    width: int = 16,
    use_bias: bool = True,
):
  num_res_blocks = (depth - 2) // 6
  if (depth - 2) % 6 != 0:
    raise ValueError('depth must be 6n+2 (e.g. 20, 32, 44).')
  
  def forward(batch, is_training):
    num_filters = width
    x, _ = batch
    x = x.astype(jnp.float32)
    x = _resnet_layer(
        x, num_filters=num_filters, activation=jax.nn.relu, use_bias=use_bias)
    for stack in range(3):
      for res_block in range(num_res_blocks):
        strides = 1
        if stack > 0 and res_block == 0:  # first layer but not first stack
          strides = 2  # downsample
        y = _resnet_layer(
            x, num_filters=num_filters, strides=strides, activation=jax.nn.relu,
            use_bias=use_bias, is_training=is_training)
        y = _resnet_layer(y, num_filters=num_filters, use_bias=use_bias)
        if stack > 0 and res_block == 0:  # first layer but not first stack
          # linear projection residual shortcut connection to match changed dims
          x = _resnet_layer(
              x, num_filters=num_filters, kernel_size=1, strides=strides,
              use_bias=use_bias)
        x = jax.nn.relu(x + y)
      num_filters *= 2
    x = hk.AvgPool(8, 8, 'VALID')(x)
    x = hk.Flatten()(x)
    logits = hk.Linear(num_classes, w_init=he_normal)(x)
    return logits
  return forward


def make_resnet20_fn(num_classes):
  return make_resnet_fn(num_classes, depth=20)


def make_cnn_lstm(num_classes,
                  max_features=20000,
                  embedding_size=128,
                  cell_size=128,
                  num_filters=64,
                  kernel_size=5,
                  pool_size=4):
  """CNN LSTM architecture for the IMDB dataset."""
  def forward(batch, is_training):
    x, _ = batch
    batch_size = x.shape[0]
    x = hk.Embed(vocab_size=max_features, embed_dim=embedding_size)(x)
    x = hk.Conv1D(output_channels=num_filters, kernel_shape=kernel_size,
                  padding="VALID")(x)
    x = jax.nn.relu(x)
    x = hk.MaxPool(
        window_shape=pool_size, strides=pool_size, padding='VALID',
        channel_axis=1)(x)
    x = jnp.moveaxis(x, 1, 0)[:, :] #[T, B, F]
    lstm_layer = hk.LSTM(hidden_size=cell_size)
    init_state = lstm_layer.initial_state(batch_size)
    # TODO: choose static vs dynamic unroll?
    x, state = hk.static_unroll(lstm_layer, x, init_state)
    x = x[-1]
    logits = hk.Linear(num_classes)(x)
    return logits
  
  return forward

  
def get_model(model_name, num_classes):
  _MODEL_FNS = {
    "lenet": make_lenet_fn,
    "resnet20": make_resnet20_fn,
    "cnn_lstm": make_cnn_lstm
  }
  net_fn = _MODEL_FNS[model_name](num_classes)
  net = hk.transform_with_state(net_fn)
  net_apply = net.apply
  net_apply = jax.experimental.callback.rewrite(
    net_apply,
    precision_utils.HIGH_PRECISION_RULES)
  return net_apply, net.init
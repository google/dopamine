# coding=utf-8
# Copyright 2023 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Networks for offline RL agents."""

import collections
from typing import Tuple

from absl import logging
from dopamine.discrete_domains import atari_lib
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp


NetworkType = collections.namedtuple('network', ['q_values', 'representation'])


def preprocess_atari_inputs(x):
  """Input normalization for Atari 2600 input frames."""
  return x.astype(jnp.float32) / 255.0


def transform_and_concat_return(x, return_to_condition):
  """Transforms and concatenates the return to the image."""
  transformed_reward = jnp.log(jnp.abs(return_to_condition) + 1) / 20.0
  transformed_reward *= jnp.sign(return_to_condition)
  tiled_reward = jnp.tile(transformed_reward, [*x.shape[:-1], 1])
  return jnp.concatenate([x, tiled_reward], axis=-1)


@gin.configurable
class Stack(nn.Module):
  """Stack of pooling and convolutional blocks with residual connections."""

  num_ch: int
  num_blocks: int
  use_max_pooling: bool = True

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    conv_out = nn.Conv(
        features=self.num_ch,
        kernel_size=(3, 3),
        strides=1,
        kernel_init=initializer,
        padding='SAME',
    )(x)
    if self.use_max_pooling:
      conv_out = nn.max_pool(
          conv_out, window_shape=(3, 3), padding='SAME', strides=(2, 2)
      )

    for _ in range(self.num_blocks):
      block_input = conv_out
      conv_out = nn.relu(conv_out)
      conv_out = nn.Conv(
          features=self.num_ch, kernel_size=(3, 3), strides=1, padding='SAME'
      )(conv_out)
      conv_out = nn.relu(conv_out)
      conv_out = nn.Conv(
          features=self.num_ch, kernel_size=(3, 3), strides=1, padding='SAME'
      )(conv_out)
      conv_out += block_input

    return conv_out


@gin.configurable
class CNNEncoder(nn.Module):
  """Convolutional DQN-inspired encoder for Atari games."""

  nn_scale: int
  conv_channels: Tuple[int, ...] = (32, 64, 64)
  conv_kernels: Tuple[int, ...] = (8, 4, 3)
  conv_strides: Tuple[int, ...] = (4, 2, 1)

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    for channel, ks_size, stride_size in zip(
        self.conv_channels, self.conv_kernels, self.conv_strides
    ):
      x = nn.Conv(
          features=channel * self.nn_scale,
          kernel_size=(ks_size, ks_size),
          strides=(stride_size, stride_size),
          kernel_init=initializer,
      )(x)
      x = nn.relu(x)
    return x


@gin.configurable
class ImpalaEncoder(nn.Module):
  """Impala Network which also outputs penultimate representation layers."""

  nn_scale: int = 1
  stack_sizes: Tuple[int, ...] = (16, 32, 32)
  num_blocks: int = 2

  def setup(self):
    self._stacks = [
        Stack(num_ch=stack_size * self.nn_scale, num_blocks=self.num_blocks)
        for stack_size in self.stack_sizes
    ]

  @nn.compact
  def __call__(self, x):
    for stack in self._stacks:
      x = stack(x)
    return nn.relu(x)


@gin.configurable
class ImpalaNetworkWithRepresentations(nn.Module):
  """Impala Network which also outputs penultimate representation layers."""

  num_actions: int
  inputs_preprocessed: bool = False
  normalize_representation: bool = False

  def setup(self):
    self.encoder = ImpalaEncoder(nn_scale=self.nn_scale)

  @nn.compact
  def __call__(self, x, stop_grad_representation=False):
    # This method sets up the MLP for inference, using the specified number of
    # layers and units.
    logging.info(
        (
            'Creating Impala network with %d nn_scale, %s stack_sizes, %d '
            'num_blocks, normalize_representation %s'
        ),
        self.nn_scale,
        str(self.stack_sizes),
        self.num_blocks,
        self.normalize_representation,
    )
    initializer = nn.initializers.xavier_uniform()
    if not self.inputs_preprocessed:
      x = preprocess_atari_inputs(x)

    conv_out = self.encoder(x)
    conv_out = conv_out.reshape(-1)

    conv_out = nn.Dense(features=512, kernel_init=initializer)(conv_out)
    representation = nn.relu(conv_out)
    if self.normalize_representation:
      representation /= jnp.linalg.norm(representation, ord=2)
    if stop_grad_representation:
      representation = jax.lax.stop_gradient(representation)
    q_values = nn.Dense(features=self.num_actions, kernel_init=initializer)(
        representation
    )
    return NetworkType(q_values, representation)


@gin.configurable
class JAXDQNNetworkWithRepresentations(nn.Module):
  """The convolutional network used to compute the agent's Q-values."""

  num_actions: int
  inputs_preprocessed: bool = False

  @nn.compact
  def __call__(self, x, stop_grad_representation=False):
    initializer = nn.initializers.xavier_uniform()
    if not self.inputs_preprocessed:
      x = preprocess_atari_inputs(x)
    logging.info('Creating Nature DQN network')
    x = nn.Conv(
        features=32,
        kernel_size=(8, 8),
        strides=(4, 4),
        kernel_init=initializer)(
            x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64,
        kernel_size=(4, 4),
        strides=(2, 2),
        kernel_init=initializer)(
            x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=initializer)(
            x)
    x = nn.relu(x)
    x = x.reshape(-1)  # flatten
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    representation = nn.relu(x)  # Use penultimate layer as representation
    if stop_grad_representation:
      representation = jax.lax.stop_gradient(representation)
    q_values = nn.Dense(
        features=self.num_actions, kernel_init=initializer)(
            representation)
    return NetworkType(q_values, representation)


@gin.configurable
class ParameterizedRainbowNetwork(nn.Module):
  """Jax Rainbow network for Full Rainbow.

  Attributes:
    num_actions: int, number of actions the agent can take at any state.
    num_atoms: int, the number of buckets of the value function distribution.
    noisy: bool, Whether to use noisy networks.
    dueling: bool, Whether to use dueling network architecture.
    distributional: bool, whether to use distributional RL.
  """

  num_actions: int
  num_atoms: int
  dueling: bool = True
  noisy: bool = False  # No exploration in offline RL, kept for compatibility.
  distributional: bool = True
  inputs_preprocessed: bool = False
  feature_dim: int = 512
  use_impala_encoder: bool = False
  nn_scale: int = 1

  def setup(self):
    if self.use_impala_encoder:
      self.encoder = ImpalaEncoder(nn_scale=self.nn_scale)
    else:
      self.encoder = CNNEncoder(nn_scale=self.nn_scale)

  @nn.compact
  def __call__(
      self, x, support, key=None, return_to_condition=None, eval_mode=False
  ):
    del key, eval_mode
    if not self.inputs_preprocessed:
      x = preprocess_atari_inputs(x)
    if return_to_condition is not None:
      x = transform_and_concat_return(x, return_to_condition)

    initializer = nn.initializers.xavier_uniform()
    x = self.encoder(x)
    x = x.reshape((-1))  # flatten
    # Single hidden layer of size feature_dim
    x = nn.Dense(
        features=self.feature_dim * self.nn_scale, kernel_init=initializer
    )(x)
    x = nn.relu(x)

    if self.dueling:
      adv = nn.Dense(features=self.num_actions * self.num_atoms)(x)
      value = nn.Dense(features=self.num_atoms)(x)
      adv = adv.reshape((self.num_actions, self.num_atoms))
      value = value.reshape((1, self.num_atoms))
      logits = value + (adv - (jnp.mean(adv, axis=0, keepdims=True)))
    else:
      x = nn.Dense(features=self.num_actions * self.num_atoms)(x)
      logits = x.reshape((self.num_actions, self.num_atoms))

    if self.distributional:
      probabilities = nn.softmax(logits)
      q_values = jnp.sum(support * probabilities, axis=1)
      return atari_lib.RainbowNetworkType(q_values, logits, probabilities)
    q_values = jnp.sum(logits, axis=1)  # Sum over all the num_atoms
    return atari_lib.DQNNetworkType(q_values)

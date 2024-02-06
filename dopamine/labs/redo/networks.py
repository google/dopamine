# coding=utf-8
# Copyright 2023 ReDo authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Defines the networks used by the RL agents."""
import math

from dopamine.discrete_domains import atari_lib
from flax import linen as nn
import gin
import jax.numpy as jnp


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return x


@gin.configurable
class ScalableNatureDQNNetwork(nn.Module):
  """The convolutional network used to compute the agent's Q-values."""

  num_actions: int
  width: int = 1
  layer_names = []

  def _record_activations(self, x, layer):
    if self.is_initializing():
      name = '/'.join(layer.scope.path)
      self.layer_names.append(name)
    return IdentityLayer(name=f'{layer.name}_act')(x)

  @nn.compact
  def __call__(self, x):
    # We need to reset the list otherwise it would have the previous values each
    # time we create a new network.
    if self.is_initializing():
      for _ in range(len(self.layer_names)):
        self.layer_names.pop()

    def _scale_width(n):
      return int(math.ceil(n * self.width))

    initializer = nn.initializers.xavier_uniform()
    # TODO(evcu) maybe remove this
    x = x.astype(jnp.float32) / 255.0
    features = (32, 64, 64)
    kernel_sizes = (8, 4, 3)
    strides = (4, 2, 1)

    for n_feature, kernel_size, stride in zip(features, kernel_sizes, strides):
      layer = nn.Conv(
          features=_scale_width(n_feature),
          kernel_size=(kernel_size, kernel_size),
          strides=(stride, stride),
          kernel_init=initializer,
      )
      x = layer(x)
      x = nn.relu(x)
      x = self._record_activations(x, layer)

    x = x.reshape((-1))  # flatten
    layer = nn.Dense(features=_scale_width(512), kernel_init=initializer)
    x = layer(x)
    x = nn.relu(x)
    x = self._record_activations(x, layer)
    layer = nn.Dense(
        features=self.num_actions, kernel_init=initializer, name='final_layer'
    )
    q_values = layer(x)
    q_values = self._record_activations(q_values, layer)
    return atari_lib.DQNNetworkType(q_values)


class Stack(nn.Module):
  """Stack of pooling and convolutional blocks with residual connections."""

  num_ch: int
  num_blocks: int = 2
  layer_names = []

  def _record_activations(self, x, layer):
    if self.is_initializing():
      name = '/'.join(layer.scope.path)
      self.layer_names.append(name)
    return IdentityLayer(name=f'{layer.name}_act')(x)

  @nn.compact
  def __call__(self, x):
    # We need to reset the list otherwise it would have the previous values each
    # time we create a new network.
    if self.is_initializing():
      for _ in range(len(self.layer_names)):
        self.layer_names.pop()
    initializer = nn.initializers.xavier_uniform()
    layer = nn.Conv(
        features=self.num_ch,
        kernel_size=(3, 3),
        strides=1,
        kernel_init=initializer,
        padding='SAME',
    )
    conv_out = layer(x)
    conv_out = nn.max_pool(
        conv_out, window_shape=(3, 3), padding='SAME', strides=(2, 2)
    )
    for _ in range(self.num_blocks):
      block_input = conv_out
      conv_out = nn.relu(conv_out)
      conv_out = self._record_activations(conv_out, layer)
      layer = nn.Conv(
          features=self.num_ch, kernel_size=(3, 3), strides=1, padding='SAME'
      )
      conv_out = layer(conv_out)
      conv_out = nn.relu(conv_out)
      conv_out = self._record_activations(conv_out, layer)
      layer = nn.Conv(
          features=self.num_ch, kernel_size=(3, 3), strides=1, padding='SAME'
      )
      conv_out = layer(conv_out)
      conv_out += block_input
    return conv_out


class ScalableDQNResNet(nn.Module):
  """ResNet used to compute the agent's Q-values."""

  num_actions: int
  width: int
  layer_names = []

  def _record_activations(self, x, layer):
    if self.is_initializing():
      name = '/'.join(layer.scope.path)
      self.layer_names.append(name)
    return IdentityLayer(name=f'{layer.name}_act')(x)

  @nn.compact
  def __call__(self, x):
    # We need to reset the list otherwise it would have the previous values each
    # time we create a new network.
    if self.is_initializing():
      for _ in range(len(self.layer_names)):
        self.layer_names.pop()

    def _scale_width(n):
      return int(math.ceil(n * self.width))

    initializer = nn.initializers.xavier_uniform()

    x = x.astype(jnp.float32) / 255.0
    for stack_size in [32, 64, 64]:
      stack = Stack(num_ch=stack_size * self.width)
      x = stack(x)
      if self.is_initializing():
        self.layer_names.extend(stack.layer_names)
    # TODO(evcu) make this safer. Each call adds new layer names to the
    # `stack.layer_names` since layer_names is treated as a class attribute
    # since it is mutable. Better way to do this would be to overwrite the
    # __init__ method to create instance objects.
    # Following will add all layer names in all stacks in order.

    x = nn.relu(x)
    x = x.reshape((-1))  # flatten
    layer = nn.Dense(features=_scale_width(512), kernel_init=initializer)
    x = layer(x)
    x = nn.relu(x)
    x = self._record_activations(x, layer)
    layer = nn.Dense(
        features=self.num_actions, kernel_init=initializer, name='final_layer'
    )
    q_values = layer(x)
    q_values = self._record_activations(q_values, layer)
    return atari_lib.DQNNetworkType(q_values)


### Rainbow Networks ###
class ScalableRainbowNetwork(nn.Module):
  """Convolutional network used to compute the agent's return distributions."""

  num_actions: int
  num_atoms: int
  width: int
  inputs_preprocessed: bool = False
  layer_names = []

  def _record_activations(self, x, layer):
    if self.is_initializing():
      name = '/'.join(layer.scope.path)
      self.layer_names.append(name)
    return IdentityLayer(name=f'{layer.name}_act')(x)

  @nn.compact
  def __call__(self, x, support):
    # We need to reset the list otherwise it would have the previous values each
    # time we create a new network.
    if self.is_initializing():
      for _ in range(len(self.layer_names)):
        self.layer_names.pop()

    def _scale_width(n):
      return int(math.ceil(n * self.width))

    initializer = nn.initializers.variance_scaling(
        scale=1.0 / jnp.sqrt(3.0), mode='fan_in', distribution='uniform'
    )
    if not self.inputs_preprocessed:
      x = x.astype(jnp.float32) / 255.0
    features = (32, 64, 64)
    kernel_sizes = (8, 4, 3)
    strides = (4, 2, 1)

    for n_feature, kernel_size, stride in zip(features, kernel_sizes, strides):
      layer = nn.Conv(
          features=_scale_width(n_feature),
          kernel_size=(kernel_size, kernel_size),
          strides=(stride, stride),
          kernel_init=initializer,
      )
      x = layer(x)
      x = nn.relu(x)
      x = self._record_activations(x, layer)

    x = x.reshape((-1))  # flatten
    layer = nn.Dense(features=_scale_width(512), kernel_init=initializer)
    x = layer(x)
    x = nn.relu(x)
    x = self._record_activations(x, layer)
    layer = nn.Dense(
        features=self.num_actions * self.num_atoms,
        kernel_init=initializer,
        name='final_layer',
    )
    x = layer(x)
    x = self._record_activations(x, layer)
    logits = x.reshape((self.num_actions, self.num_atoms))
    probabilities = nn.softmax(logits)
    q_values = jnp.sum(support * probabilities, axis=1)
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)


@gin.configurable
class FullRainbowNetwork(nn.Module):
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
  noisy: bool = True
  dueling: bool = True
  distributional: bool = True
  inputs_preprocessed: bool = False
  layer_names = []

  def _record_activations(self, x, layer):
    if self.is_initializing():
      name = '/'.join(layer.scope.path)
      self.layer_names.append(name)
    return IdentityLayer(name=f'{layer.name}_act')(x)

  @nn.compact
  def __call__(self, x, support, eval_mode=False, key=None):
    # Generate a random number generation key if not provided
    # if key is None:
    #   key = jax.random.PRNGKey(int(time.time() * 1e6))

    if not self.inputs_preprocessed:
      x = x.astype(jnp.float32) / 255.0

    hidden_sizes = [32, 64, 64]
    kernel_sizes = [8, 4, 3]
    stride_sizes = [4, 2, 1]
    for hidden_size, kernel_size, stride_size in zip(
        hidden_sizes, kernel_sizes, stride_sizes
    ):
      layer = nn.Conv(
          features=hidden_size,
          kernel_size=(kernel_size, kernel_size),
          strides=(stride_size, stride_size),
          kernel_init=nn.initializers.xavier_uniform(),
      )
      x = layer(x)
      x = nn.relu(x)
      x = self._record_activations(x, layer)
    x = x.reshape((-1))  # flatten

    layer = nn.Dense(features=512, kernel_init=nn.initializers.xavier_uniform())
    x = layer(x)
    x = nn.relu(x)
    x = self._record_activations(x, layer)

    if self.dueling:
      layer = nn.Dense(
          features=self.num_actions * self.num_atoms,
          kernel_init=nn.initializers.xavier_uniform(),
      )
      adv = layer(x)
      adv = self._record_activations(adv, layer)
      layer = nn.Dense(
          features=self.num_atoms, kernel_init=nn.initializers.xavier_uniform()
      )
      value = layer(x)
      value = self._record_activations(value, layer)
      adv = adv.reshape((self.num_actions, self.num_atoms))
      value = value.reshape((1, self.num_atoms))
      logits = value + (adv - (jnp.mean(adv, axis=0, keepdims=True)))
    else:
      layer = nn.Dense(
          features=self.num_actions * self.num_atoms,
          kernel_init=nn.initializers.xavier_uniform(),
      )
      x = layer(x)
      logits = x.reshape((self.num_actions, self.num_atoms))
      logits = self._record_activations(logits, layer)

    if self.distributional:
      probabilities = nn.softmax(logits)
      q_values = jnp.sum(support * probabilities, axis=1)
      return atari_lib.RainbowNetworkType(q_values, logits, probabilities)
    q_values = jnp.sum(logits, axis=1)  # Sum over all the num_atoms
    return atari_lib.DQNNetworkType(q_values)

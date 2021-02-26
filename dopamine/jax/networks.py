# coding=utf-8
# Copyright 2018 The Dopamine Authors.
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
"""Various networks for Jax Dopamine agents."""

from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import gym_lib
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import numpy as onp


gin.constant('jax_networks.CARTPOLE_OBSERVATION_DTYPE', jnp.float64)
gin.constant('jax_networks.ACROBOT_OBSERVATION_DTYPE', jnp.float64)
gin.constant('jax_networks.LUNAR_OBSERVATION_DTYPE', jnp.float64)
gin.constant('jax_networks.MOUNTAINCAR_OBSERVATION_DTYPE', jnp.float64)


### DQN Networks ###
@gin.configurable
class NatureDQNNetwork(nn.Module):
  """The convolutional network used to compute the agent's Q-values."""
  num_actions: int

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    x = x.astype(jnp.float32) / 255.
    x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = x.reshape((-1))  # flatten
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    q_values = nn.Dense(features=self.num_actions,
                        kernel_init=initializer)(x)
    return atari_lib.DQNNetworkType(q_values)


# TODO(psc): Consolidate the classic control networks to avoid code duplication.
@gin.configurable
class CartpoleDQNNetwork(nn.Module):
  """Jax DQN network for Cartpole."""
  num_actions: int

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    x = x.astype(jnp.float32)
    x = x.reshape((-1))  # flatten
    x -= gym_lib.CARTPOLE_MIN_VALS
    x /= gym_lib.CARTPOLE_MAX_VALS - gym_lib.CARTPOLE_MIN_VALS
    x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    q_values = nn.Dense(features=self.num_actions, kernel_init=initializer)(x)
    return atari_lib.DQNNetworkType(q_values)


@gin.configurable
class AcrobotDQNNetwork(nn.Module):
  """Jax DQN network for Acrobot."""
  num_actions: int

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    x = x.astype(jnp.float32)
    x = x.reshape((-1))  # flatten
    x -= gym_lib.ACROBOT_MIN_VALS
    x /= gym_lib.ACROBOT_MAX_VALS - gym_lib.ACROBOT_MIN_VALS
    x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    q_values = nn.Dense(features=self.num_actions, kernel_init=initializer)(x)
    return atari_lib.DQNNetworkType(q_values)


@gin.configurable
class LunarLanderDQNNetwork(nn.Module):
  """Jax DQN network for LunarLander."""
  num_actions: int

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    x = x.astype(jnp.float32)
    x = x.reshape((-1))  # flatten
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    q_values = nn.Dense(features=self.num_actions, kernel_init=initializer)(x)
    return atari_lib.DQNNetworkType(q_values)


@gin.configurable
class MountainCarDQNNetwork(nn.Module):
  """Jax DQN network for MountainCar."""
  num_actions: int

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    x = x.astype(jnp.float32)
    x = x.reshape((-1))  # flatten
    x -= gym_lib.MOUNTAINCAR_MIN_VALS
    x /= gym_lib.MOUNTAINCAR_MAX_VALS - gym_lib.MOUNTAINCAR_MIN_VALS
    x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    q_values = nn.Dense(features=self.num_actions, kernel_init=initializer)(x)
    return atari_lib.DQNNetworkType(q_values)


### Rainbow Networks ###
def softmax_cross_entropy_loss_with_logits(labels, logits):
  return -jnp.sum(labels * nn.log_softmax(logits))


@gin.configurable
class RainbowNetwork(nn.Module):
  """Convolutional network used to compute the agent's return distributions."""
  num_actions: int
  num_atoms: int

  @nn.compact
  def __call__(self, x, support):
    initializer = nn.initializers.variance_scaling(
        scale=1.0 / jnp.sqrt(3.0),
        mode='fan_in',
        distribution='uniform')
    x = x.astype(jnp.float32) / 255.
    x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = x.reshape((-1))  # flatten
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.num_actions * self.num_atoms,
                 kernel_init=initializer)(x)
    logits = x.reshape((self.num_actions, self.num_atoms))
    probabilities = nn.softmax(logits)
    q_values = jnp.sum(support * probabilities, axis=1)
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)


@gin.configurable
class CartpoleRainbowNetwork(nn.Module):
  """Jax Rainbow network for Cartpole."""
  num_actions: int
  num_atoms: int

  @nn.compact
  def __call__(self, x, support):
    initializer = nn.initializers.xavier_uniform()
    x = x.astype(jnp.float32)
    x = x.reshape((-1))  # flatten
    x -= gym_lib.CARTPOLE_MIN_VALS
    x /= gym_lib.CARTPOLE_MAX_VALS - gym_lib.CARTPOLE_MIN_VALS
    x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.num_actions * self.num_atoms,
                 kernel_init=initializer)(x)
    logits = x.reshape((self.num_actions, self.num_atoms))
    probabilities = nn.softmax(logits)
    q_values = jnp.sum(support * probabilities, axis=1)
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)


@gin.configurable
class AcrobotRainbowNetwork(nn.Module):
  """Jax Rainbow network for Acrobot."""
  num_actions: int
  num_atoms: int

  @nn.compact
  def __call__(self, x, support):
    initializer = nn.initializers.xavier_uniform()
    x = x.astype(jnp.float32)
    x = x.reshape((-1))  # flatten
    x -= gym_lib.ACROBOT_MIN_VALS
    x /= gym_lib.ACROBOT_MAX_VALS - gym_lib.ACROBOT_MIN_VALS
    x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.num_actions * self.num_atoms,
                 kernel_init=initializer)(x)
    logits = x.reshape((self.num_actions, self.num_atoms))
    probabilities = nn.softmax(logits)
    q_values = jnp.sum(support * probabilities, axis=1)
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)


### Implicit Quantile Networks ###
class ImplicitQuantileNetwork(nn.Module):
  """The Implicit Quantile Network (Dabney et al., 2018).."""
  num_actions: int
  quantile_embedding_dim: int

  @nn.compact
  def __call__(self, x, num_quantiles, rng):
    initializer = nn.initializers.variance_scaling(
        scale=1.0 / jnp.sqrt(3.0),
        mode='fan_in',
        distribution='uniform')
    x = x.astype(jnp.float32) / 255.
    x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = x.reshape((-1))  # flatten
    state_vector_length = x.shape[-1]
    state_net_tiled = jnp.tile(x, [num_quantiles, 1])
    quantiles_shape = [num_quantiles, 1]
    quantiles = jax.random.uniform(rng, shape=quantiles_shape)
    quantile_net = jnp.tile(quantiles, [1, self.quantile_embedding_dim])
    quantile_net = (
        jnp.arange(1, self.quantile_embedding_dim + 1, 1).astype(jnp.float32)
        * onp.pi
        * quantile_net)
    quantile_net = jnp.cos(quantile_net)
    quantile_net = nn.Dense(features=state_vector_length,
                            kernel_init=initializer)(quantile_net)
    quantile_net = nn.relu(quantile_net)
    x = state_net_tiled * quantile_net
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    quantile_values = nn.Dense(features=self.num_actions,
                               kernel_init=initializer)(x)
    return atari_lib.ImplicitQuantileNetworkType(quantile_values, quantiles)


### Quantile Networks ###
@gin.configurable
class QuantileNetwork(nn.Module):
  """Convolutional network used to compute the agent's return quantiles."""
  num_actions: int
  num_atoms: int

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.variance_scaling(
        scale=1.0 / jnp.sqrt(3.0),
        mode='fan_in',
        distribution='uniform')
    x = x.astype(jnp.float32) / 255.
    x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = x.reshape((-1))  # flatten
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.num_actions * self.num_atoms,
                 kernel_init=initializer)(x)
    logits = x.reshape((self.num_actions, self.num_atoms))
    probabilities = nn.softmax(logits)
    q_values = jnp.mean(logits, axis=1)
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)

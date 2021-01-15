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
from flax import nn
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

  def apply(self, x, num_actions):
    initializer = nn.initializers.xavier_uniform()
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    x = x[None, ...]
    x = x.astype(jnp.float32) / 255.
    x = nn.Conv(x, features=32, kernel_size=(8, 8), strides=(4, 4),
                kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Conv(x, features=64, kernel_size=(4, 4), strides=(2, 2),
                kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Conv(x, features=64, kernel_size=(3, 3), strides=(1, 1),
                kernel_init=initializer)
    x = jax.nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    q_values = nn.Dense(x, features=num_actions, kernel_init=initializer)
    return atari_lib.DQNNetworkType(q_values)


@gin.configurable
class CartpoleDQNNetwork(nn.Module):
  """Jax DQN network for Cartpole."""

  def apply(self, x, num_actions):
    initializer = nn.initializers.xavier_uniform()
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = x.reshape((x.shape[0], -1))  # flatten
    x -= gym_lib.CARTPOLE_MIN_VALS
    x /= gym_lib.CARTPOLE_MAX_VALS - gym_lib.CARTPOLE_MIN_VALS
    x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    q_values = nn.Dense(x, features=num_actions, kernel_init=initializer)
    return atari_lib.DQNNetworkType(q_values)


@gin.configurable
class AcrobotDQNNetwork(nn.Module):
  """Jax DQN network for Cartpole."""

  def apply(self, x, num_actions):
    initializer = nn.initializers.xavier_uniform()
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = x.reshape((x.shape[0], -1))  # flatten
    x -= gym_lib.ACROBOT_MIN_VALS
    x /= gym_lib.ACROBOT_MAX_VALS - gym_lib.ACROBOT_MIN_VALS
    x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    q_values = nn.Dense(x, features=num_actions, kernel_init=initializer)
    return atari_lib.DQNNetworkType(q_values)


@gin.configurable
class LunarLanderDQNNetwork(nn.Module):
  """Jax DQN network for LunarLander."""

  def apply(self, x, num_actions):
    initializer = nn.initializers.xavier_uniform()
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    q_values = nn.Dense(x, features=num_actions, kernel_init=initializer)
    return atari_lib.DQNNetworkType(q_values)


@gin.configurable
class MountainCarDQNNetwork(nn.Module):
  """Jax DQN network for MountainCar."""

  def apply(self, x, num_actions):
    initializer = nn.initializers.xavier_uniform()
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = x.reshape((x.shape[0], -1))  # flatten
    x -= gym_lib.MOUNTAINCAR_MIN_VALS
    x /= gym_lib.MOUNTAINCAR_MAX_VALS - gym_lib.MOUNTAINCAR_MIN_VALS
    x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    q_values = nn.Dense(x, features=num_actions, kernel_init=initializer)
    return atari_lib.DQNNetworkType(q_values)


### Rainbow Networks ###
def softmax_cross_entropy_loss_with_logits(labels, logits):
  return -jnp.sum(labels * jax.nn.log_softmax(logits))


@gin.configurable
class RainbowNetwork(nn.Module):
  """Convolutional network used to compute the agent's return distributions."""

  def apply(self, x, num_actions, num_atoms, support):
    initializer = jax.nn.initializers.variance_scaling(
        scale=1.0 / jnp.sqrt(3.0),
        mode='fan_in',
        distribution='uniform')
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    x = x[None, ...]
    x = x.astype(jnp.float32) / 255.
    x = nn.Conv(x, features=32, kernel_size=(8, 8), strides=(4, 4),
                kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Conv(x, features=64, kernel_size=(4, 4), strides=(2, 2),
                kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Conv(x, features=64, kernel_size=(3, 3), strides=(1, 1),
                kernel_init=initializer)
    x = jax.nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Dense(x, features=num_actions * num_atoms, kernel_init=initializer)
    logits = x.reshape((x.shape[0], num_actions, num_atoms))
    probabilities = nn.softmax(logits)
    q_values = jnp.sum(support * probabilities, axis=2)
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)


@gin.configurable
class CartpoleRainbowNetwork(nn.Module):
  """Jax Rainbow network for Cartpole."""

  def apply(self, x, num_actions, num_atoms, support):
    initializer = nn.initializers.xavier_uniform()
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = x.reshape((x.shape[0], -1))  # flatten
    x -= gym_lib.CARTPOLE_MIN_VALS
    x /= gym_lib.CARTPOLE_MAX_VALS - gym_lib.CARTPOLE_MIN_VALS
    x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Dense(x, features=num_actions * num_atoms, kernel_init=initializer)
    logits = x.reshape((x.shape[0], num_actions, num_atoms))
    probabilities = nn.softmax(logits)
    q_values = jnp.sum(support * probabilities, axis=2)
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)


@gin.configurable
class AcrobotRainbowNetwork(nn.Module):
  """Jax Rainbow network for Acrobot."""

  def apply(self, x, num_actions, num_atoms, support):
    initializer = nn.initializers.xavier_uniform()
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    x = x[None, ...]
    x = x.astype(jnp.float32)
    x = x.reshape((x.shape[0], -1))  # flatten
    x -= gym_lib.ACROBOT_MIN_VALS
    x /= gym_lib.ACROBOT_MAX_VALS - gym_lib.ACROBOT_MIN_VALS
    x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Dense(x, features=num_actions * num_atoms, kernel_init=initializer)
    logits = x.reshape((x.shape[0], num_actions, num_atoms))
    probabilities = nn.softmax(logits)
    q_values = jnp.sum(support * probabilities, axis=2)
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)


### Implicit Quantile Networks ###
class ImplicitQuantileNetwork(nn.Module):
  """The Implicit Quantile Network (Dabney et al., 2018).."""

  def apply(self, x, num_actions, quantile_embedding_dim, num_quantiles, rng):
    initializer = jax.nn.initializers.variance_scaling(
        scale=1.0 / jnp.sqrt(3.0),
        mode='fan_in',
        distribution='uniform')
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    x = x[None, ...]
    x = x.astype(jnp.float32) / 255.
    x = nn.Conv(x, features=32, kernel_size=(8, 8), strides=(4, 4),
                kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Conv(x, features=64, kernel_size=(4, 4), strides=(2, 2),
                kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Conv(x, features=64, kernel_size=(3, 3), strides=(1, 1),
                kernel_init=initializer)
    x = jax.nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    state_vector_length = x.shape[-1]
    state_net_tiled = jnp.tile(x, [num_quantiles, 1])
    quantiles_shape = [num_quantiles, 1]
    quantiles = jax.random.uniform(rng, shape=quantiles_shape)
    quantile_net = jnp.tile(quantiles, [1, quantile_embedding_dim])
    quantile_net = (
        jnp.arange(1, quantile_embedding_dim + 1, 1).astype(jnp.float32)
        * onp.pi
        * quantile_net)
    quantile_net = jnp.cos(quantile_net)
    quantile_net = nn.Dense(quantile_net,
                            features=state_vector_length,
                            kernel_init=initializer)
    quantile_net = jax.nn.relu(quantile_net)
    x = state_net_tiled * quantile_net
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    quantile_values = nn.Dense(x, features=num_actions, kernel_init=initializer)
    return atari_lib.ImplicitQuantileNetworkType(quantile_values, quantiles)


### Quantile Networks ###
@gin.configurable
class QuantileNetwork(nn.Module):
  """Convolutional network used to compute the agent's return quantiles."""

  def apply(self, x, num_actions, num_atoms):
    initializer = jax.nn.initializers.variance_scaling(
        scale=1.0 / jnp.sqrt(3.0),
        mode='fan_in',
        distribution='uniform')
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    x = x[None, ...]
    x = x.astype(jnp.float32) / 255.
    x = nn.Conv(x, features=32, kernel_size=(8, 8), strides=(4, 4),
                kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Conv(x, features=64, kernel_size=(4, 4), strides=(2, 2),
                kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Conv(x, features=64, kernel_size=(3, 3), strides=(1, 1),
                kernel_init=initializer)
    x = jax.nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Dense(x, features=num_actions * num_atoms, kernel_init=initializer)
    logits = x.reshape((x.shape[0], num_actions, num_atoms))
    probabilities = nn.softmax(logits)
    q_values = jnp.mean(logits, axis=2)
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)

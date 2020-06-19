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


gin.constant('jax_networks.CARTPOLE_OBSERVATION_DTYPE', jnp.float64)
gin.constant('jax_networks.ACROBOT_OBSERVATION_DTYPE', jnp.float64)


@gin.configurable
class NatureDQNNetwork(nn.Module):
  """The convolutional network used to compute the agent's Q-values."""

  def apply(self, x, num_actions):
    initializer = nn.initializers.xavier_uniform()
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

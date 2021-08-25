# coding=utf-8
# Copyright 2021 The Dopamine Authors.
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
"""Networks for continuous control agents."""

import functools
import operator
from typing import NamedTuple, Optional, Tuple

from flax import linen as nn
import gin
import jax
from jax import numpy as jnp
import tensorflow as tf
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class SacActorOutput(NamedTuple):
  """The output of a SAC actor."""
  mean_action: jnp.ndarray
  sampled_action: jnp.ndarray
  log_probability: jnp.ndarray


class SacCriticOutput(NamedTuple):
  """The output of a SAC critic."""
  q_value1: jnp.ndarray
  q_value2: jnp.ndarray


class SacOutput(NamedTuple):
  """The output of a SACNetwork, including the actor and critic outputs."""
  actor: SacActorOutput
  critic: SacCriticOutput


class _Tanh(tfb.Tanh):

  def _inverse(self, y):
    # We perform clipping in the _inverse function, as is done in TF-Agents.
    y = jnp.where(
        jnp.less_equal(jnp.abs(y), 1.), tf.clip(y, -0.99999997, 0.99999997), y)
    return jnp.arctanh(y)


def _transform_distribution(dist, mean, magnitude):
  """Scales the input normal distribution to be within the action limits.

  Args:
    dist: a TensorFlow distribution.
    mean: desired action means.
    magnitude: desired action magnitudes.

  Returns:
    A transformed distribution, scaled to within the action limits.
  """
  bijectors = tfb.Chain([
      tfb.Shift(mean)(tfb.Scale(magnitude)),
      _Tanh(),
  ])
  dist = tfd.TransformedDistribution(dist, bijectors)
  return dist


def _shifted_uniform(minval=0., maxval=1.0, dtype=jnp.float32):

  def init(key, shape, dtype=dtype):
    return jax.random.uniform(
        key, shape=shape, minval=minval, maxval=maxval, dtype=dtype)

  return init


class SACCriticNetwork(nn.Module):
  """A simple critic network used in SAC."""
  num_layers: int = 2
  hidden_units: int = 256

  @nn.compact
  def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    kernel_initializer = jax.nn.initializers.glorot_uniform()

    # Preprocess inputs
    a = action.reshape(-1)  # flatten
    x = state.astype(jnp.float32)
    x = x.reshape(-1)  # flatten
    x = jnp.concatenate((x, a))

    for _ in range(self.num_layers):
      x = nn.Dense(
          features=self.hidden_units, kernel_init=kernel_initializer)(
              x)
      x = nn.relu(x)

    return nn.Dense(features=1, kernel_init=kernel_initializer)(x)


@gin.configurable
class SACNetwork(nn.Module):
  """Non-convolutional value and policy networks for SAC."""
  action_shape: Tuple[int, ...]
  num_layers: int = 2
  hidden_units: int = 256
  action_limits: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None

  def setup(self):
    action_dim = functools.reduce(operator.mul, self.action_shape, 1)

    # The setting of these initializers were borrowed from the TF-Agents SAC
    # implementation.
    kernel_initializer = jax.nn.initializers.glorot_uniform()

    self._critic1 = SACCriticNetwork(self.num_layers, self.hidden_units)
    self._critic2 = SACCriticNetwork(self.num_layers, self.hidden_units)

    self._actor_layers = [
        nn.Dense(features=self.hidden_units, kernel_init=kernel_initializer)
        for _ in range(self.num_layers)
    ]
    self._actor_final_layer = nn.Dense(
        features=action_dim * 2, kernel_init=kernel_initializer)

  def __call__(self,
               state: jnp.ndarray,
               key: jnp.ndarray,
               mean_action: bool = True) -> SacOutput:
    """Calls the SAC actor/critic networks.

    This has two important purposes:
      1. It is used to initialize all parameters of both networks.
      2. It is used to efficiently calculate the outputs of both the actor
        and critic networks on a single input.

    Args:
      state: An input state.
      key: A PRNGKey to use to sample an action from the actor's output
        distribution.
      mean_action: If True, it will use the actor's mean action to feed to the
        value network. Otherwise, it will use the sampled action.

    Returns:
      A named tuple containing the outputs from both networks.
    """
    actor_output = self.actor(state, key)

    if mean_action:
      critic_output = self.critic(state, actor_output.mean_action)
    else:
      critic_output = self.critic(state, actor_output.sampled_action)

    return SacOutput(actor_output, critic_output)

  def actor(self, state: jnp.ndarray, key: jnp.ndarray) -> SacActorOutput:
    """Calls the SAC actor network.

    This can be called using network_def.apply(..., method=network_def.actor).

    Args:
      state: An input state.
      key: A PRNGKey to use to sample an action from the actor's output
        distribution.

    Returns:
      A named tuple containing a sampled action, the mean action, and the
        likelihood of the sampled action.
    """
    # Preprocess inputs
    x = state.astype(jnp.float32)
    x = x.reshape(-1)  # flatten

    for layer in self._actor_layers:
      x = layer(x)
      x = nn.relu(x)

    # Note we are only producing a diagonal covariance matrix, not a full
    # covariance matrix as it is difficult to ensure that it would be PSD.
    loc_and_scale_diag = self._actor_final_layer(x)
    loc, scale_diag = jnp.split(loc_and_scale_diag, 2)
    # Exponentiate to only get positive terms.
    scale_diag = jnp.exp(scale_diag)
    dist = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)
    if self.action_limits is None:
      mode = dist.mode()
    else:
      lower_action_limit = jnp.asarray(self.action_limits[0], dtype=jnp.float32)
      upper_action_limit = jnp.asarray(self.action_limits[1], dtype=jnp.float32)
      mean = (lower_action_limit + upper_action_limit) / 2.0
      magnitude = (upper_action_limit - lower_action_limit) / 2.0
      mode = magnitude * jnp.tanh(dist.mode()) + mean
      dist = _transform_distribution(dist, mean, magnitude)
    sampled_action = dist.sample(seed=key)
    action_probability = dist.log_prob(sampled_action)

    mode = jnp.reshape(mode, self.action_shape)
    sampled_action = jnp.reshape(sampled_action, self.action_shape)

    return SacActorOutput(mode, sampled_action, action_probability)

  def critic(self, state: jnp.ndarray, action: jnp.ndarray) -> SacCriticOutput:
    """Calls the SAC critic network.

    SAC uses two Q networks to reduce overestimation bias.
    This can be called using network_def.apply(..., method=network_def.critic).

    Args:
      state: An input state.
      action: An action to compute the value of.

    Returns:
      A named tuple containing the Q values of each network.
    """
    return SacCriticOutput(
        self._critic1(state, action), self._critic2(state, action))

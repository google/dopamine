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

from collections.abc import Callable
import functools
import operator
from typing import NamedTuple, Optional, Tuple

from flax import linen as nn
import gin
import jax
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class ActorOutput(NamedTuple):
  """The output of a continuous actor."""

  mean_action: jnp.ndarray
  sampled_action: jnp.ndarray
  log_probability: jnp.ndarray


class CriticOutput(NamedTuple):
  """The output of a continuous critic."""

  q_value1: jnp.ndarray
  q_value2: jnp.ndarray


class ActorCriticOutput(NamedTuple):
  """The output of the actor critic network."""

  actor: ActorOutput
  critic: CriticOutput


class PPOActorOutput(NamedTuple):
  """The output of a continuous ppo actor."""

  sampled_action: jnp.ndarray
  log_probability: jnp.ndarray
  entropy: jnp.ndarray


class PPOCriticOutput(NamedTuple):
  """The output of a continuous ppo critic."""

  q_value: jnp.ndarray


class PPOActorCriticOutput(NamedTuple):
  """The output of the ppo actor critic network."""

  actor: PPOActorOutput
  critic: PPOCriticOutput


class _Tanh(tfb.Tanh):

  def _inverse(self, y):
    # We perform clipping in the _inverse function, as is done in TF-Agents.
    y = jnp.where(
        jnp.less_equal(jnp.abs(y), 1.0), jnp.clip(y, -0.99999997, 0.99999997), y
    )
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


def _shifted_uniform(minval=0.0, maxval=1.0, dtype=jnp.float32):
  def init(key, shape, dtype=dtype):
    return jax.random.uniform(
        key, shape=shape, minval=minval, maxval=maxval, dtype=dtype
    )

  return init


@gin.configurable
def create_activation(
    activation='relu',
):
  """Create an activation function for the network."""
  if activation == 'relu':
    return nn.relu
  elif activation == 'tanh':
    return nn.tanh
  else:
    raise ValueError('Unsupported activation {}'.format(activation))


class ActorNetwork(nn.Module):
  """A simple continuous actor network."""

  action_shape: Tuple[int, ...]
  num_layers: int = 2
  hidden_units: int = 256
  action_limits: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None
  activation: Callable[[jax.Array], jax.Array] = nn.relu
  kernel_initializer: jax.nn.initializers.Initializer = (
      jax.nn.initializers.glorot_uniform()
  )

  @nn.compact
  def __call__(
      self,
      state: jnp.ndarray,
      key: Optional[jnp.ndarray] = None,
      action: Optional[jnp.ndarray] = None,
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    action_dim = functools.reduce(operator.mul, self.action_shape, 1)

    # Preprocess inputs
    x = state.astype(jnp.float32)
    x = x.reshape(-1)  # flatten

    for _ in range(self.num_layers):
      x = nn.Dense(
          features=self.hidden_units, kernel_init=self.kernel_initializer
      )(x)
      x = self.activation(x)

    # Note we are only producing a diagonal covariance matrix, not a full
    # covariance matrix as it is difficult to ensure that it would be PSD.
    loc_and_scale_diag = nn.Dense(
        features=action_dim * 2, kernel_init=self.kernel_initializer
    )(x)
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
    if action is None:
      if key is None:
        raise ValueError('Key must be provided if action is None.')
      action = dist.sample(seed=key)
    action_probability = dist.log_prob(action)

    mode = jnp.reshape(mode, self.action_shape)
    action = jnp.reshape(action, self.action_shape)

    return mode, action, action_probability


class CriticNetwork(nn.Module):
  """A simple continuous critic network."""

  num_layers: int = 2
  hidden_units: int = 256
  activation: Callable[[jax.Array], jax.Array] = nn.relu
  kernel_initializer: jax.nn.initializers.Initializer = (
      jax.nn.initializers.glorot_uniform()
  )

  @nn.compact
  def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:

    # Preprocess inputs
    a = action.reshape(-1)  # flatten
    x = state.astype(jnp.float32)
    x = x.reshape(-1)  # flatten
    x = jnp.concatenate((x, a))

    for _ in range(self.num_layers):
      x = nn.Dense(
          features=self.hidden_units, kernel_init=self.kernel_initializer
      )(x)
      x = self.activation(x)

    return nn.Dense(features=1, kernel_init=self.kernel_initializer)(x)


@gin.configurable
class ActorCriticNetwork(nn.Module):
  """Non-convolutional actor critic value and policy networks."""

  action_shape: Tuple[int, ...]
  num_layers: int = 2
  hidden_units: int = 256
  action_limits: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None
  activation: Callable[[jax.Array], jax.Array] = nn.relu
  kernel_initializer: jax.nn.initializers.Initializer = (
      jax.nn.initializers.glorot_uniform()
  )

  def setup(self):
    self._actor = ActorNetwork(
        self.action_shape,
        self.num_layers,
        self.hidden_units,
        self.action_limits,
        self.activation,
        self.kernel_initializer,
    )
    self._critic1 = CriticNetwork(
        self.num_layers,
        self.hidden_units,
        self.activation,
        self.kernel_initializer,
    )
    self._critic2 = CriticNetwork(
        self.num_layers,
        self.hidden_units,
        self.activation,
        self.kernel_initializer,
    )

  def __call__(
      self, state: jnp.ndarray, key: jnp.ndarray, mean_action: bool = True
  ) -> ActorCriticOutput:
    """Calls the actor/critic networks.

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

    return ActorCriticOutput(actor_output, critic_output)

  def actor(self, state: jnp.ndarray, key: jnp.ndarray) -> ActorOutput:
    """Calls the actor network.

    This can be called using network_def.apply(..., method=network_def.actor).

    Args:
      state: An input state.
      key: A PRNGKey to use to sample an action from the actor's output
        distribution.

    Returns:
      A named tuple containing a sampled action, the mean action, and the
        likelihood of the sampled action.
    """
    return ActorOutput(*self._actor(state, key=key))

  def critic(self, state: jnp.ndarray, action: jnp.ndarray) -> CriticOutput:
    """Calls the critic network.

    SAC uses two Q networks to reduce overestimation bias.
    This can be called using network_def.apply(..., method=network_def.critic).

    Args:
      state: An input state.
      action: An action to compute the value of.

    Returns:
      A named tuple containing the Q values of each network.
    """
    return CriticOutput(
        self._critic1(state, action), self._critic2(state, action)
    )


@gin.configurable
class PPOActorNetwork(nn.Module):
  """Actor network for PPO.

  Attributes:
    action_shape: shape of the action.
    num_layers: number of layers in the network.
    hidden_units: number of hidden units in the network.
    activation: activation function to use in the network.
  """

  action_shape: Tuple[int, ...]
  num_layers: int = 2
  hidden_units: int = 64
  action_limits: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None
  activation: Callable[[jax.Array], jax.Array] = nn.tanh

  @nn.compact
  def __call__(
      self,
      state: jnp.ndarray,
      key: Optional[jnp.ndarray] = None,
      action: Optional[jnp.ndarray] = None,
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    action_dim = functools.reduce(operator.mul, self.action_shape, 1)

    # Preprocess inputs
    x = state.astype(jnp.float32)
    x = x.reshape(-1)  # flatten

    for _ in range(self.num_layers):
      x = nn.Dense(
          features=self.hidden_units,
          kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
      )(x)
      x = self.activation(x)
    loc = nn.Dense(
        features=action_dim,
        kernel_init=nn.initializers.orthogonal(jnp.sqrt(0.01)),
    )(x)
    # scale_diag is state independent and initialized to zero.
    scale_diag = nn.Dense(
        features=action_dim, kernel_init=nn.initializers.zeros
    )(jnp.ones_like(x))
    scale_diag = jnp.exp(scale_diag)

    dist = tfd.Normal(loc=loc, scale=scale_diag)

    # Compute the entropy before transforming, as the transformation does not
    # have a constant Jacobian.
    entropy = jnp.sum(dist.entropy())
    if self.action_limits is not None:
      lower_action_limit = jnp.asarray(self.action_limits[0], dtype=jnp.float32)
      upper_action_limit = jnp.asarray(self.action_limits[1], dtype=jnp.float32)
      mean = (lower_action_limit + upper_action_limit) / 2.0
      magnitude = (upper_action_limit - lower_action_limit) / 2.0
      dist = _transform_distribution(dist, mean, magnitude)

    if action is None:
      if key is None:
        raise ValueError('Key must be provided if action is None.')
      action = dist.sample(seed=key)

    log_probability = jnp.sum(dist.log_prob(action))

    action = jnp.reshape(action, self.action_shape)

    return action, log_probability, entropy


class PPOCriticNetwork(nn.Module):
  """Critic network for PPO.

  Attributes:
    num_layers: number of layers in the network.
    hidden_units: number of hidden units in the network.
    activation: activation function to use in the network.
  """

  num_layers: int = 2
  hidden_units: int = 64
  activation: Callable[[jax.Array], jax.Array] = nn.tanh

  @nn.compact
  def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
    # Preprocess inputs
    x = state.astype(jnp.float32)
    x = x.reshape(-1)  # flatten

    for _ in range(self.num_layers):
      x = nn.Dense(
          features=self.hidden_units,
          kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
      )(x)
      x = self.activation(x)
    return nn.Dense(features=1, kernel_init=nn.initializers.orthogonal(1.0))(x)


@gin.configurable
class PPOActorCriticNetwork(nn.Module):
  """Actor critic network for PPO.

  Attributes:
    action_shape: shape of the action.
    num_layers: number of layers in the networks.
    hidden_units: number of hidden units in the networks.
    activation: activation function to use in the networks.
  """

  action_shape: Tuple[int, ...]
  num_layers: int = 2
  hidden_units: int = 64
  action_limits: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None
  activation: Callable[[jax.Array], jax.Array] = nn.tanh

  def setup(self):
    self._actor = PPOActorNetwork(
        self.action_shape,
        self.num_layers,
        self.hidden_units,
        self.action_limits,
        self.activation,
    )
    self._critic = PPOCriticNetwork(
        self.num_layers, self.hidden_units, self.activation
    )

  def __call__(
      self, state: jnp.ndarray, key: jnp.ndarray
  ) -> PPOActorCriticOutput:
    """Calls the actor/critic networks.

    This has two important purposes:
      1. It is used to initialize all parameters of both networks.
      2. It is used to efficiently calculate the outputs of both the actor
        and critic networks on a single input.

    Args:
      state: An input state.
      key: A PRNGKey to use to sample an action from the actor's output
        distribution.

    Returns:
      A named tuple containing the outputs from both networks.
    """
    actor_output = self.actor(state, key)
    critic_output = self.critic(state)
    return PPOActorCriticOutput(actor_output, critic_output)

  def actor(
      self,
      state: jnp.ndarray,
      key: Optional[jnp.ndarray] = None,
      action: Optional[jnp.ndarray] = None,
  ) -> PPOActorOutput:
    """Calls the actor network.

    This can be called using network_def.apply(..., method=network_def.actor).

    Args:
      state: An input state.
      key: A PRNGKey to use to sample an action from the actor's output
        distribution.
      action: An action to use to feed to the actor network.

    Returns:
      A named tuple containing the provided action, the likelihood of the
      provided action, and the entropy of the action.
    """
    return PPOActorOutput(*self._actor(state, key, action))

  def critic(self, state: jnp.ndarray) -> PPOCriticOutput:
    """Calls the critic network.

    This can be called using network_def.apply(..., method=network_def.critic).

    Args:
      state: An input state.

    Returns:
      A named tuple containing the Q value of the network.
    """
    return PPOCriticOutput(self._critic(state))

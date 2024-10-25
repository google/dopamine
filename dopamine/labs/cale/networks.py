# coding=utf-8
# Copyright 2024 The Dopamine Authors.
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
"""Networks for running SAC/PPO on CALE."""

from typing import Callable, Optional, Tuple

from absl import logging
from dopamine.jax import continuous_networks
from dopamine.jax import networks as discrete_networks
from dopamine.labs.sac_from_pixels import continuous_networks as pixel_continuous_networks
from flax import linen as nn
import gin
import jax
from jax import numpy as jnp


@gin.configurable
class NatureDQNEncoder(nn.Module):
  """An encoder based on the original DQN network."""

  projection_dimension: int = 512

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    # Images are stored in the replay buffer as uint8.
    x = x.astype(jnp.float32) / 255.0
    x = nn.Conv(
        features=32, kernel_size=(8, 8), strides=(4, 4), kernel_init=initializer
    )(x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64, kernel_size=(4, 4), strides=(2, 2), kernel_init=initializer
    )(x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64, kernel_size=(3, 3), strides=(1, 1), kernel_init=initializer
    )(x)
    x = nn.relu(x)
    x = jnp.reshape(x, -1)  # Flatten

    critic_z = nn.Dense(
        features=self.projection_dimension, kernel_init=initializer
    )(x)
    critic_z = nn.LayerNorm()(critic_z)
    critic_z = nn.relu(critic_z)

    # Only the critic should train the convolution layers, so stop the
    # gradients from the actor.
    actor_z = nn.Dense(
        features=self.projection_dimension, kernel_init=initializer
    )(jax.lax.stop_gradient(x))
    actor_z = nn.relu(actor_z)

    return pixel_continuous_networks.SACEncoderOutputs(critic_z, actor_z)


@gin.configurable
class SACImpalaEncoder(nn.Module):
  """An encoder based on the Impala encoder."""

  projection_dimension: int = 512
  nn_scale: int = 1
  stack_sizes: Tuple[int, ...] = (16, 32, 32)
  num_blocks: int = 2

  def setup(self):
    logging.info('\t Creating %s ...', self.__class__.__name__)
    logging.info('\t projection_dimension: %s', self.projection_dimension)
    logging.info('\t num_blocks: %s', self.num_blocks)
    logging.info('\t nn_scale: %s', self.nn_scale)
    logging.info('\t stack_sizes: %s', self.stack_sizes)
    self._stacks = [
        discrete_networks.Stack(
            num_ch=stack_size * self.nn_scale, num_blocks=self.num_blocks
        )
        for stack_size in self.stack_sizes
    ]

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    for stack in self._stacks:
      x = stack(x)

    critic_z = nn.Dense(
        features=self.projection_dimension, kernel_init=initializer
    )(x)
    critic_z = nn.LayerNorm()(critic_z)
    critic_z = nn.relu(critic_z)

    # Only the critic should train the convolution layers, so stop the
    # gradients from the actor.
    actor_z = nn.Dense(
        features=self.projection_dimension, kernel_init=initializer
    )(jax.lax.stop_gradient(x))
    actor_z = nn.relu(actor_z)

    return pixel_continuous_networks.SACEncoderOutputs(critic_z, actor_z)


@gin.configurable
class SACCALEConvNetwork(nn.Module):
  """A convolutional value and policy networks for SAC/PPO."""

  action_shape: Tuple[int, ...]
  num_layers: int = 2
  hidden_units: int = 256
  action_limits: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None
  encoder_name: str = 'NatureDQN'
  sac_cls: nn.Module = continuous_networks.ActorCriticNetwork

  def setup(self):
    logging.info(
        '\t Creating %s with encoder %s...',
        self.__class__.__name__,
        self.encoder_name,
    )
    if self.encoder_name == 'NatureDQN':
      self._encoder = NatureDQNEncoder()
    elif self.encoder_name == 'Impala':
      self._encoder = SACImpalaEncoder()
    elif self.encoder_name == 'SAC':
      self._encoder = pixel_continuous_networks.SACEncoderNetwork()
    else:
      raise ValueError(f'Unrecognized encoder: {self.encoder_name}')

    self._sac_network = self.sac_cls(
        self.action_shape,
        self.num_layers,
        self.hidden_units,
        self.action_limits,
    )

  def __call__(
      self, state: jnp.ndarray, key: jnp.ndarray, mean_action: bool = True
  ) -> continuous_networks.ActorCriticOutput:
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
    encoding = self._encoder(state)

    actor_output = self._sac_network.actor(encoding.actor_z, key)
    action = (
        actor_output.mean_action if mean_action else actor_output.sampled_action
    )
    critic_output = self._sac_network.critic(encoding.critic_z, action)

    return continuous_networks.ActorCriticOutput(actor_output, critic_output)

  def actor(
      self, state: jnp.ndarray, key: jnp.ndarray
  ) -> continuous_networks.ActorOutput:
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
    encoding = self._encoder(state)
    return self._sac_network.actor(encoding.actor_z, key)

  def critic(
      self, state: jnp.ndarray, action: jnp.ndarray
  ) -> continuous_networks.CriticOutput:
    """Calls the SAC critic network.

    SAC uses two Q networks to reduce overestimation bias.
    This can be called using network_def.apply(..., method=network_def.critic).

    Args:
      state: An input state.
      action: An action to compute the value of.

    Returns:
      A named tuple containing the Q values of each network.
    """
    encoding = self._encoder(state)
    return self._sac_network.critic(encoding.critic_z, action)


@gin.configurable
class PPOCALEConvNetwork(nn.Module):
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
  activation: Callable[[jax.Array], jax.Array] = nn.tanh
  action_limits: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None
  encoder_name: str = 'NatureDQN'
  ppo_cls: nn.Module = continuous_networks.PPOActorCriticNetwork

  def setup(self):
    logging.info(
        '\t Creating %s with encoder %s...',
        self.__class__.__name__,
        self.encoder_name,
    )
    if self.encoder_name == 'NatureDQN':
      self._encoder = NatureDQNEncoder()
    elif self.encoder_name == 'Impala':
      self._encoder = SACImpalaEncoder()
    elif self.encoder_name == 'SAC':
      self._encoder = pixel_continuous_networks.SACEncoderNetwork()
    else:
      raise ValueError(f'Unrecognized encoder: {self.encoder_name}')
    self._ppo_network = self.ppo_cls(
        action_shape=self.action_shape,
        action_limits=self.action_limits,
        num_layers=self.num_layers,
        hidden_units=self.hidden_units,
    )

  def __call__(
      self, state: jnp.ndarray, key: jnp.ndarray
  ) -> continuous_networks.PPOActorCriticOutput:
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
    encoding = self._encoder(state)
    actor_output = self._ppo_network.actor(encoding.actor_z, key)
    critic_output = self._ppo_network.critic(encoding.critic_z)
    return continuous_networks.PPOActorCriticOutput(actor_output, critic_output)

  def actor(
      self,
      state: jnp.ndarray,
      key: Optional[jnp.ndarray] = None,
      action: Optional[jnp.ndarray] = None,
  ) -> continuous_networks.PPOActorOutput:
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
    encoding = self._encoder(state)
    return self._ppo_network.actor(encoding.actor_z, key=key, action=action)

  def critic(self, state: jnp.ndarray) -> continuous_networks.PPOCriticOutput:
    """Calls the critic network.

    This can be called using network_def.apply(..., method=network_def.critic).

    Args:
      state: An input state.

    Returns:
      A named tuple containing the Q value of the network.
    """
    encoding = self._encoder(state)
    return self._ppo_network.critic(encoding.critic_z)

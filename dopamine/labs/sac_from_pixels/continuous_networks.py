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

import dataclasses
from typing import Optional, Tuple

from dopamine.jax import continuous_networks
from flax import linen as nn
import gin
import jax
from jax import numpy as jnp


@dataclasses.dataclass
class SACEncoderOutputs:
  """The output of a SACEncoder."""

  critic_z: jnp.ndarray
  actor_z: jnp.ndarray


class SACEncoderNetwork(nn.Module):
  """An encoder network for soft actor critic.

  This is the network described in the SAC-AE paper "Improving sample
  efficiency in model-free reinforcement learning from images"
  (Yarats, Zhang, Kostrikov, Amos, Pineau, & Fergus, 2019), and used in
  experiments in the DrQ paper "Image Augmentation Is All You Need:
  Regularizing Deep Reinforcement Learning from Pixels" (Kostrikov, Yarats,
  & Fergus, 2020).
  """

  @nn.compact
  def __call__(self, x):
    # Images are stored in the replay buffer as uint8.
    x = x.astype(jnp.float32) / 255.0

    # Flatten the last dimension (normally to deal with stacked rgb frames)
    if len(x.shape) > 3:
      x = x.reshape((*x.shape[:2], -1))

    kernel_init = nn.initializers.orthogonal()
    x = nn.Conv(
        features=32, kernel_size=(3, 3), strides=(2, 2), kernel_init=kernel_init
    )(x)
    x = nn.relu(x)
    x = nn.Conv(
        features=32, kernel_size=(3, 3), strides=(1, 1), kernel_init=kernel_init
    )(x)
    x = nn.relu(x)
    x = nn.Conv(
        features=32, kernel_size=(3, 3), strides=(1, 1), kernel_init=kernel_init
    )(x)
    x = nn.relu(x)
    x = nn.Conv(
        features=32, kernel_size=(3, 3), strides=(1, 1), kernel_init=kernel_init
    )(x)
    x = nn.relu(x)
    x = jnp.reshape(x, -1)  # Flatten

    critic_z = nn.Dense(features=50, kernel_init=kernel_init)(x)
    critic_z = nn.LayerNorm()(critic_z)
    critic_z = nn.tanh(critic_z)

    # Only the critic should train the convolution layers, so stop the
    # gradients from the actor.
    actor_z = nn.Dense(features=50, kernel_init=kernel_init)(
        jax.lax.stop_gradient(x)
    )
    actor_z = nn.LayerNorm()(actor_z)
    actor_z = nn.tanh(actor_z)

    return SACEncoderOutputs(critic_z, actor_z)


@gin.configurable
class SACConvNetwork(nn.Module):
  """A convolutional value and policy networks for SAC.

  This uses the SAC-AE network for processing images. For more information,
  view SACEncoderNetwork.
  """

  action_shape: Tuple[int, ...]
  num_layers: int = 2
  hidden_units: int = 256
  action_limits: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None

  def setup(self):
    self._encoder = SACEncoderNetwork()
    self._sac_network = continuous_networks.ActorCriticNetwork(
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

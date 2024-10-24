# coding=utf-8
# Copyright 2021 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#      http://www.apache.org/licenses/LICENSE-2.0
#
# This code is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES 
# OR CONDITIONS OF ANY KIND, either express or implied. See the 
# License for the specific language governing permissions and
# limitations under the License.
"""
Defines network models for continuous control agents using Soft Actor-Critic (SAC).
"""

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
    """Represents the output of a Soft Actor-Critic (SAC) actor."""
    mean_action: jnp.ndarray
    sampled_action: jnp.ndarray
    log_probability: jnp.ndarray


class SacCriticOutput(NamedTuple):
    """Represents the output of a Soft Actor-Critic (SAC) critic."""
    q_value1: jnp.ndarray
    q_value2: jnp.ndarray


class SacOutput(NamedTuple):
    """Represents the combined output of SAC actor and critic networks."""
    actor: SacActorOutput
    critic: SacCriticOutput


class _TanhBijector(tfb.Tanh):
    """Custom Tanh bijector with clipping to ensure numerical stability."""

    def _inverse(self, y):
        # Clip values within the range [-0.99999997, 0.99999997] to avoid numerical instability.
        y = jnp.where(
            jnp.abs(y) <= 1.0, 
            tf.clip_by_value(y, -0.99999997, 0.99999997), 
            y
        )
        return jnp.arctanh(y)


def transform_distribution(dist, mean, magnitude):
    """
    Scales the input normal distribution to fit within the specified action limits.

    Args:
        dist: A TensorFlow probability distribution.
        mean: Desired action means.
        magnitude: Desired action magnitudes.

    Returns:
        A transformed distribution scaled to within the action limits.
    """
    bijectors = tfb.Chain([
        tfb.Shift(mean)(tfb.Scale(magnitude)),
        _TanhBijector(),
    ])
    return tfd.TransformedDistribution(dist, bijectors)


def shifted_uniform(minval=0.0, maxval=1.0, dtype=jnp.float32):
    """
    Initializes a uniform distribution with given minimum and maximum values.
    """
    def init(key, shape, dtype=dtype):
        return jax.random.uniform(key, shape=shape, minval=minval, maxval=maxval, dtype=dtype)
    return init


class SACCriticNetwork(nn.Module):
    """Defines the critic network used in Soft Actor-Critic (SAC) models."""

    num_layers: int = 2
    hidden_units: int = 256

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of the critic network.

        Args:
            state: Current state input.
            action: Action input.

        Returns:
            The estimated Q-value.
        """
        kernel_init = jax.nn.initializers.glorot_uniform()

        # Flatten the input state and action
        action_flat = action.reshape(-1)
        state_flat = state.astype(jnp.float32).reshape(-1)
        x = jnp.concatenate([state_flat, action_flat])

        # Apply fully connected layers
        for _ in range(self.num_layers):
            x = nn.Dense(features=self.hidden_units, kernel_init=kernel_init)(x)
            x = nn.relu(x)

        return nn.Dense(features=1, kernel_init=kernel_init)(x)


@gin.configurable
class SACNetwork(nn.Module):
    """
    Defines a Soft Actor-Critic (SAC) network with separate actor and critic networks.
    """
    action_shape: Tuple[int, ...]
    num_layers: int = 2
    hidden_units: int = 256
    action_limits: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None

    def setup(self):
        """Initializes the SAC network components."""
        action_dim = functools.reduce(operator.mul, self.action_shape, 1)
        kernel_init = jax.nn.initializers.glorot_uniform()

        # Define the critic networks
        self.critic1 = SACCriticNetwork(self.num_layers, self.hidden_units)
        self.critic2 = SACCriticNetwork(self.num_layers, self.hidden_units)

        # Define the actor network layers
        self.actor_layers = [
            nn.Dense(features=self.hidden_units, kernel_init=kernel_init) for _ in range(self.num_layers)
        ]
        self.actor_final_layer = nn.Dense(features=action_dim * 2, kernel_init=kernel_init)

    def __call__(self, state: jnp.ndarray, key: jnp.ndarray, mean_action: bool = True) -> SacOutput:
        """
        Forward pass for both actor and critic networks.

        Args:
            state: Input state.
            key: PRNG key for action sampling.
            mean_action: Whether to use mean action or sample from the distribution.

        Returns:
            The combined output from both actor and critic networks.
        """
        actor_output = self.actor(state, key)
        critic_output = self.critic(state, actor_output.mean_action if mean_action else actor_output.sampled_action)
        return SacOutput(actor=actor_output, critic=critic_output)

    def actor(self, state: jnp.ndarray, key: jnp.ndarray) -> SacActorOutput:
        """
        Forward pass for the actor network.

        Args:
            state: Input state.
            key: PRNG key for action sampling.

        Returns:
            The actor's action and log-probability output.
        """
        x = state.astype(jnp.float32).reshape(-1)

        # Apply the actor's hidden layers
        for layer in self.actor_layers:
            x = layer(x)
            x = nn.relu(x)

        # Produce mean and scale for the action distribution
        loc_and_scale_diag = self.actor_final_layer(x)
        loc, scale_diag = jnp.split(loc_and_scale_diag, 2)
        scale_diag = jnp.exp(scale_diag)
        dist = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        # Transform the action distribution if action limits are provided
        if self.action_limits is not None:
            mean = (jnp.array(self.action_limits[0]) + jnp.array(self.action_limits[1])) / 2.0
            magnitude = (jnp.array(self.action_limits[1]) - jnp.array(self.action_limits[0])) / 2.0
            mode = magnitude * jnp.tanh(dist.mode()) + mean
            dist = transform_distribution(dist, mean, magnitude)
        else:
            mode = dist.mode()

        sampled_action = dist.sample(seed=key)
        action_probability = dist.log_prob(sampled_action)

        return SacActorOutput(mean_action=mode.reshape(self.action_shape), 
                              sampled_action=sampled_action.reshape(self.action_shape),
                              log_probability=action_probability)

    def critic(self, state: jnp.ndarray, action: jnp.ndarray) -> SacCriticOutput:
        """
        Forward pass for the critic network, producing Q-values for the given action.

        Args:
            state: Input state.
            action: Action to evaluate.

        Returns:
            The Q-values from both critic networks.
        """
        return SacCriticOutput(
            q_value1=self.critic1(state, action),
            q_value2=self.critic2(state, action)
        )

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
"""Legacy (pre-Keras) network architectures."""

from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import gym_lib
import gin


### ATARI networks ###


def nature_dqn_network(num_actions, network_type, state):
  """The convolutional network used to compute the agent's Q-values.

  Args:
    num_actions: int, number of actions.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  model = atari_lib.NatureDQNNetwork(num_actions, network_type)
  net = model(state)
  return network_type(net.q_values)


def rainbow_network(num_actions, num_atoms, support, network_type, state):
  """The convolutional network used to compute agent's Q-value distributions.

  Args:
    num_actions: int, number of actions.
    num_atoms: int, the number of buckets of the value function distribution.
    support: tf.linspace, the support of the Q-value distribution.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  model = atari_lib.RainbowNetwork(num_actions, num_atoms, support)
  net = model(state)
  return network_type(net.q_values, net.logits, net.probabilities)


def implicit_quantile_network(
    num_actions, quantile_embedding_dim, network_type, state, num_quantiles
):
  """The Implicit Quantile ConvNet.

  Args:
    num_actions: int, number of actions.
    quantile_embedding_dim: int, embedding dimension for the quantile input.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.
    num_quantiles: int, number of quantile inputs.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  model = atari_lib.ImplicitQuantileNetwork(num_actions, quantile_embedding_dim)
  net = model(state, num_quantiles)
  return network_type(
      quantile_values=net.quantile_values, quantiles=net.quantiles
  )


### Generic Gym networks ###


@gin.configurable
def _basic_discrete_domain_network(
    min_vals, max_vals, num_actions, state, num_atoms=None
):
  """Builds a basic network for discrete domains, rescaling inputs to [-1, 1].

  Args:
    min_vals: float, minimum attainable values (must be same shape as `state`).
    max_vals: float, maximum attainable values (must be same shape as `state`).
    num_actions: int, number of actions.
    state: `tf.Tensor`, the state input.
    num_atoms: int or None, if None will construct a DQN-style network,
      otherwise will construct a Rainbow-style network.

  Returns:
    The Q-values for DQN-style agents or logits for Rainbow-style agents.
  """
  layer = gym_lib.BasicDiscreteDomainNetwork(
      min_vals, max_vals, num_actions, num_atoms
  )
  return layer(state)


@gin.configurable
def cartpole_dqn_network(num_actions, network_type, state):
  """Builds the deep network used to compute the agent's Q-values.

  It rescales the input features to a range that yields improved performance.

  Args:
    num_actions: int, number of actions.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  model = gym_lib.CartpoleDQNNetwork(num_actions)
  net = model(state)
  return network_type(net.q_values)


@gin.configurable
def fourier_dqn_network(
    min_vals, max_vals, num_actions, state, fourier_basis_order=3
):
  """Builds the function approximator used to compute the agent's Q-values.

  It uses FourierBasis features and a linear layer.

  Args:
    min_vals: float, minimum attainable values (must be same shape as `state`).
    max_vals: float, maximum attainable values (must be same shape as `state`).
    num_actions: int, number of actions.
    state: `tf.Tensor`, contains the agent's current state.
    fourier_basis_order: int, order of the Fourier basis functions.

  Returns:
    The Q-values for DQN-style agents or logits for Rainbow-style agents.
  """
  model = gym_lib.FourierDQNNetwork(
      min_vals, max_vals, num_actions, fourier_basis_order
  )
  return model(state).q_values


def cartpole_fourier_dqn_network(num_actions, network_type, state):
  """Builds the function approximator used to compute the agent's Q-values.

  It uses the Fourier basis features and a linear function approximator.

  Args:
    num_actions: int, number of actions.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  model = gym_lib.CartpoleFourierDQNNetwork(num_actions)
  net = model(state)
  return network_type(net.q_values)


@gin.configurable
def cartpole_rainbow_network(
    num_actions, num_atoms, support, network_type, state
):
  """Build the deep network used to compute the agent's Q-value distributions.

  Args:
    num_actions: int, number of actions.
    num_atoms: int, the number of buckets of the value function distribution.
    support: tf.linspace, the support of the Q-value distribution.
    network_type: `namedtuple`, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  model = gym_lib.CartpoleRainbowNetwork(num_actions, num_atoms, support)
  net = model(state)
  return network_type(net.q_values, net.logits, net.probabilities)


@gin.configurable
def acrobot_dqn_network(num_actions, network_type, state):
  """Builds the deep network used to compute the agent's Q-values.

  It rescales the input features to a range that yields improved performance.

  Args:
    num_actions: int, number of actions.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  model = gym_lib.AcrobotDQNNetwork(num_actions)
  net = model(state)
  return network_type(net.q_values)


@gin.configurable
def acrobot_fourier_dqn_network(num_actions, network_type, state):
  """Builds the function approximator used to compute the agent's Q-values.

  It uses the Fourier basis features and a linear function approximator.

  Args:
    num_actions: int, number of actions.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  model = gym_lib.AcrobotFourierDQNNetwork(num_actions)
  net = model(state)
  return network_type(net.q_values)


@gin.configurable
def acrobot_rainbow_network(
    num_actions, num_atoms, support, network_type, state
):
  """Build the deep network used to compute the agent's Q-value distributions.

  Args:
    num_actions: int, number of actions.
    num_atoms: int, the number of buckets of the value function distribution.
    support: tf.linspace, the support of the Q-value distribution.
    network_type: `namedtuple`, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  model = gym_lib.AcrobotRainbowNetwork(num_actions, num_atoms, support)
  net = model(state)
  return network_type(net.q_values, net.logits, net.probabilities)

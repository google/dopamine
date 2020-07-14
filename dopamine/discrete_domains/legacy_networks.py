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

import math

from dopamine.discrete_domains import gym_lib
import gin
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import slim as contrib_slim


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
  net = tf.cast(state, tf.float32)
  net = net / 255
  net = contrib_slim.conv2d(net, 32, [8, 8], stride=4)
  net = contrib_slim.conv2d(net, 64, [4, 4], stride=2)
  net = contrib_slim.conv2d(net, 64, [3, 3], stride=1)
  net = contrib_slim.flatten(net)
  net = contrib_slim.fully_connected(net, 512)
  q_values = contrib_slim.fully_connected(net, num_actions, activation_fn=None)
  return network_type(q_values)


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
  weights_initializer = contrib_slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

  net = tf.cast(state, tf.float32)
  net = net / 255
  net = contrib_slim.conv2d(
      net, 32, [8, 8], stride=4, weights_initializer=weights_initializer)
  net = contrib_slim.conv2d(
      net, 64, [4, 4], stride=2, weights_initializer=weights_initializer)
  net = contrib_slim.conv2d(
      net, 64, [3, 3], stride=1, weights_initializer=weights_initializer)
  net = contrib_slim.flatten(net)
  net = contrib_slim.fully_connected(
      net, 512, weights_initializer=weights_initializer)
  net = contrib_slim.fully_connected(
      net,
      num_actions * num_atoms,
      activation_fn=None,
      weights_initializer=weights_initializer)

  logits = tf.reshape(net, [-1, num_actions, num_atoms])
  probabilities = contrib_layers.softmax(logits)
  q_values = tf.reduce_sum(support * probabilities, axis=2)
  return network_type(q_values, logits, probabilities)


def implicit_quantile_network(num_actions, quantile_embedding_dim,
                              network_type, state, num_quantiles):
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
  weights_initializer = contrib_slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

  state_net = tf.cast(state, tf.float32)
  state_net = state_net / 255
  state_net = contrib_slim.conv2d(
      state_net, 32, [8, 8], stride=4, weights_initializer=weights_initializer)
  state_net = contrib_slim.conv2d(
      state_net, 64, [4, 4], stride=2, weights_initializer=weights_initializer)
  state_net = contrib_slim.conv2d(
      state_net, 64, [3, 3], stride=1, weights_initializer=weights_initializer)
  state_net = contrib_slim.flatten(state_net)
  state_net_size = state_net.get_shape().as_list()[-1]
  state_net_tiled = tf.tile(state_net, [num_quantiles, 1])

  batch_size = state_net.get_shape().as_list()[0]
  quantiles_shape = [num_quantiles * batch_size, 1]
  quantiles = tf.random.uniform(
      quantiles_shape, minval=0, maxval=1, dtype=tf.float32)

  quantile_net = tf.tile(quantiles, [1, quantile_embedding_dim])
  pi = tf.constant(math.pi)
  quantile_net = tf.cast(tf.range(
      1, quantile_embedding_dim + 1, 1), tf.float32) * pi * quantile_net
  quantile_net = tf.cos(quantile_net)
  quantile_net = contrib_slim.fully_connected(
      quantile_net, state_net_size, weights_initializer=weights_initializer)
  # Hadamard product.
  net = tf.multiply(state_net_tiled, quantile_net)

  net = contrib_slim.fully_connected(
      net, 512, weights_initializer=weights_initializer)
  quantile_values = contrib_slim.fully_connected(
      net,
      num_actions,
      activation_fn=None,
      weights_initializer=weights_initializer)

  return network_type(quantile_values=quantile_values, quantiles=quantiles)


### Generic Gym networks ###


@gin.configurable
def _basic_discrete_domain_network(min_vals, max_vals, num_actions, state,
                                   num_atoms=None):
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
  net = tf.cast(state, tf.float32)
  net = contrib_slim.flatten(net)
  net -= min_vals
  net /= max_vals - min_vals
  net = 2.0 * net - 1.0  # Rescale in range [-1, 1].
  net = contrib_slim.fully_connected(net, 512)
  net = contrib_slim.fully_connected(net, 512)
  if num_atoms is None:
    # We are constructing a DQN-style network.
    return contrib_slim.fully_connected(net, num_actions, activation_fn=None)
  else:
    # We are constructing a Rainbow-style network.
    return contrib_slim.fully_connected(
        net, num_actions * num_atoms, activation_fn=None)


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
  q_values = _basic_discrete_domain_network(
      gym_lib.CARTPOLE_MIN_VALS, gym_lib.CARTPOLE_MAX_VALS, num_actions, state)
  return network_type(q_values)


@gin.configurable
def fourier_dqn_network(min_vals,
                        max_vals,
                        num_actions,
                        state,
                        fourier_basis_order=3):
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
  net = tf.cast(state, tf.float32)
  net = contrib_slim.flatten(net)

  # Feed state through Fourier basis.
  feature_generator = gym_lib.FourierBasis(
      net.get_shape().as_list()[-1],
      min_vals,
      max_vals,
      order=fourier_basis_order)
  net = feature_generator.compute_features(net)

  # Q-values are always linear w.r.t. last layer.
  q_values = contrib_slim.fully_connected(
      net, num_actions, activation_fn=None, biases_initializer=None)
  return q_values


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
  q_values = fourier_dqn_network(gym_lib.CARTPOLE_MIN_VALS,
                                 gym_lib.CARTPOLE_MAX_VALS,
                                 num_actions, state)
  return network_type(q_values)


@gin.configurable
def cartpole_rainbow_network(num_actions, num_atoms, support, network_type,
                             state):
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
  net = _basic_discrete_domain_network(
      gym_lib.CARTPOLE_MIN_VALS, gym_lib.CARTPOLE_MAX_VALS, num_actions, state,
      num_atoms=num_atoms)
  logits = tf.reshape(net, [-1, num_actions, num_atoms])
  probabilities = contrib_layers.softmax(logits)
  q_values = tf.reduce_sum(support * probabilities, axis=2)
  return network_type(q_values, logits, probabilities)


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
  q_values = _basic_discrete_domain_network(
      gym_lib.ACROBOT_MIN_VALS, gym_lib.ACROBOT_MAX_VALS, num_actions, state)
  return network_type(q_values)


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
  q_values = fourier_dqn_network(gym_lib.ACROBOT_MIN_VALS,
                                 gym_lib.ACROBOT_MAX_VALS,
                                 num_actions, state)
  return network_type(q_values)


@gin.configurable
def acrobot_rainbow_network(num_actions, num_atoms, support, network_type,
                            state):
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
  net = _basic_discrete_domain_network(
      gym_lib.ACROBOT_MIN_VALS, gym_lib.ACROBOT_MAX_VALS, num_actions, state,
      num_atoms=num_atoms)
  logits = tf.reshape(net, [-1, num_actions, num_atoms])
  probabilities = contrib_layers.softmax(logits)
  q_values = tf.reduce_sum(support * probabilities, axis=2)
  return network_type(q_values, logits, probabilities)

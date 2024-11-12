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
"""Legacy (TF) network architectures."""
import itertools
import math

from absl import logging
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import gym_lib
import gin
import numpy as np
import tensorflow as tf



NATURE_DQN_OBSERVATION_SHAPE = (84, 84)  # Size of downscaled Atari 2600 frame.
NATURE_DQN_DTYPE = tf.uint8  # DType of Atari 2600 observations.
NATURE_DQN_STACK_SIZE = 4  # Number of frames in the state stack.

DQNNetworkType = atari_lib.DQNNetworkType
RainbowNetworkType = atari_lib.RainbowNetworkType
ImplicitQuantileNetworkType = atari_lib.ImplicitQuantileNetworkType


@gin.configurable(denylist=['variables'])
def maybe_transform_variable_names(variables, legacy_checkpoint_load=False):
  """Maps old variable names to the new ones.

  The resulting dictionary can be passed to the tf.compat.v1.train.Saver to load
  legacy checkpoints into Keras models.

  Args:
    variables: list, of all variables to be transformed.
    legacy_checkpoint_load: bool, if True the variable names are mapped to the
      legacy names as appeared in `tf.slim` based agents. Use this if you want
      to load checkpoints saved before tf.keras.Model upgrade.

  Returns:
    dict or None, of <new_names, var>.
  """
  logging.info('legacy_checkpoint_load: %s', legacy_checkpoint_load)
  if legacy_checkpoint_load:
    name_map = {}
    for var in variables:
      new_name = var.op.name.replace('bias', 'biases')
      new_name = new_name.replace('kernel', 'weights')
      name_map[new_name] = var
  else:
    name_map = None
  return name_map


### Keras ATARI networks ###
class NatureDQNNetwork(tf.keras.Model):
  """The convolutional network used to compute the agent's Q-values."""

  def __init__(self, num_actions, name=None):
    """Creates the layers used for calculating Q-values.

    Args:
      num_actions: int, number of actions.
      name: str, used to create scope for network parameters.
    """
    super(NatureDQNNetwork, self).__init__(name=name)

    self.num_actions = num_actions
    # Defining layers.
    activation_fn = tf.keras.activations.relu
    # Setting names of the layers manually to make variable names more similar
    # with tf.slim variable names/checkpoints.
    self.conv1 = tf.keras.layers.Conv2D(
        32,
        [8, 8],
        strides=4,
        padding='same',
        activation=activation_fn,
        name='Conv',
    )
    self.conv2 = tf.keras.layers.Conv2D(
        64,
        [4, 4],
        strides=2,
        padding='same',
        activation=activation_fn,
        name='Conv',
    )
    self.conv3 = tf.keras.layers.Conv2D(
        64,
        [3, 3],
        strides=1,
        padding='same',
        activation=activation_fn,
        name='Conv',
    )
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(
        512, activation=activation_fn, name='fully_connected'
    )
    self.dense2 = tf.keras.layers.Dense(num_actions, name='fully_connected')

  def call(self, state):
    """Creates the output tensor/op given the state tensor as input.

    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.

    Parameters created here will have scope according to the `name` argument
    given at `.__init__()` call.
    Args:
      state: Tensor, input tensor.

    Returns:
      collections.namedtuple, output ops (graph mode) or output tensors (eager).
    """
    x = tf.cast(state, tf.float32)
    x = x / 255
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.dense1(x)

    return DQNNetworkType(self.dense2(x))


class RainbowNetwork(tf.keras.Model):
  """The convolutional network used to compute agent's return distributions."""

  def __init__(self, num_actions, num_atoms, support, name=None):
    """Creates the layers used calculating return distributions.

    Args:
      num_actions: int, number of actions.
      num_atoms: int, the number of buckets of the value function distribution.
      support: tf.linspace, the support of the Q-value distribution.
      name: str, used to crete scope for network parameters.
    """
    super(RainbowNetwork, self).__init__(name=name)
    activation_fn = tf.keras.activations.relu
    self.num_actions = num_actions
    self.num_atoms = num_atoms
    self.support = support

    def kernel_initializer():
      return tf.keras.initializers.VarianceScaling(
          scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform'
      )

    # Defining layers.
    self.conv1 = tf.keras.layers.Conv2D(
        32,
        [8, 8],
        strides=4,
        padding='same',
        activation=activation_fn,
        kernel_initializer=kernel_initializer(),
        name='Conv',
    )
    self.conv2 = tf.keras.layers.Conv2D(
        64,
        [4, 4],
        strides=2,
        padding='same',
        activation=activation_fn,
        kernel_initializer=kernel_initializer(),
        name='Conv',
    )
    self.conv3 = tf.keras.layers.Conv2D(
        64,
        [3, 3],
        strides=1,
        padding='same',
        activation=activation_fn,
        kernel_initializer=kernel_initializer(),
        name='Conv',
    )
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(
        512,
        activation=activation_fn,
        kernel_initializer=kernel_initializer(),
        name='fully_connected',
    )
    self.dense2 = tf.keras.layers.Dense(
        num_actions * num_atoms,
        kernel_initializer=kernel_initializer(),
        name='fully_connected',
    )

  def call(self, state):
    """Creates the output tensor/op given the state tensor as input.

    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.

    Args:
      state: Tensor, input tensor.

    Returns:
      collections.namedtuple, output ops (graph mode) or output tensors (eager).
    """
    x = tf.cast(state, tf.float32)
    x = x / 255
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.dense1(x)
    x = self.dense2(x)
    logits = tf.reshape(x, [-1, self.num_actions, self.num_atoms])
    probabilities = tf.keras.activations.softmax(logits)
    q_values = tf.reduce_sum(self.support * probabilities, axis=2)
    return RainbowNetworkType(q_values, logits, probabilities)


class ImplicitQuantileNetwork(tf.keras.Model):
  """The Implicit Quantile Network (Dabney et al., 2018).."""

  def __init__(self, num_actions, quantile_embedding_dim, name=None):
    """Creates the layers used calculating quantile values.

    Args:
      num_actions: int, number of actions.
      quantile_embedding_dim: int, embedding dimension for the quantile input.
      name: str, used to create scope for network parameters.
    """
    super(ImplicitQuantileNetwork, self).__init__(name=name)
    self.num_actions = num_actions
    self.quantile_embedding_dim = quantile_embedding_dim
    # We need the activation function during `call`, therefore set the field.
    self.activation_fn = tf.keras.activations.relu

    def kernel_initializer():
      return tf.keras.initializers.VarianceScaling(
          scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform'
      )

    self.kernel_initializer = kernel_initializer
    # Defining layers.
    self.conv1 = tf.keras.layers.Conv2D(
        32,
        [8, 8],
        strides=4,
        padding='same',
        activation=self.activation_fn,
        kernel_initializer=kernel_initializer(),
        name='Conv',
    )
    self.conv2 = tf.keras.layers.Conv2D(
        64,
        [4, 4],
        strides=2,
        padding='same',
        activation=self.activation_fn,
        kernel_initializer=kernel_initializer(),
        name='Conv',
    )
    self.conv3 = tf.keras.layers.Conv2D(
        64,
        [3, 3],
        strides=1,
        padding='same',
        activation=self.activation_fn,
        kernel_initializer=kernel_initializer(),
        name='Conv',
    )
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(
        512,
        activation=self.activation_fn,
        kernel_initializer=kernel_initializer(),
        name='fully_connected',
    )
    self.dense2 = tf.keras.layers.Dense(
        num_actions,
        kernel_initializer=kernel_initializer(),
        name='fully_connected',
    )

  def call(self, state, num_quantiles):
    """Creates the output tensor/op given the state tensor as input.

    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.

    Args:
      state: `tf.Tensor`, contains the agent's current state.
      num_quantiles: int, number of quantile inputs.

    Returns:
      collections.namedtuple, that contains (quantile_values, quantiles).
    """
    batch_size = state.get_shape().as_list()[0]
    x = tf.cast(state, tf.float32)
    x = x / 255
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    state_vector_length = x.get_shape().as_list()[-1]
    state_net_tiled = tf.tile(x, [num_quantiles, 1])
    quantiles_shape = [num_quantiles * batch_size, 1]
    quantiles = tf.random.uniform(
        quantiles_shape, minval=0, maxval=1, dtype=tf.float32
    )
    quantile_net = tf.tile(quantiles, [1, self.quantile_embedding_dim])
    pi = tf.constant(math.pi)
    quantile_net = (
        tf.cast(tf.range(1, self.quantile_embedding_dim + 1, 1), tf.float32)
        * pi
        * quantile_net
    )
    quantile_net = tf.cos(quantile_net)
    # Create the quantile layer in the first call. This is because
    # number of output units depends on the input shape. Therefore, we can only
    # create the layer during the first forward call, not during `.__init__()`.
    if not hasattr(self, 'dense_quantile'):
      self.dense_quantile = tf.keras.layers.Dense(
          state_vector_length,
          activation=self.activation_fn,
          kernel_initializer=self.kernel_initializer(),
      )
    quantile_net = self.dense_quantile(quantile_net)
    x = tf.multiply(state_net_tiled, quantile_net)
    x = self.dense1(x)
    quantile_values = self.dense2(x)
    return ImplicitQuantileNetworkType(quantile_values, quantiles)


### Pre-keras ATARI networks ###


def nature_dqn_network(num_actions, network_type, state):
  """The convolutional network used to compute the agent's Q-values.

  Args:
    num_actions: int, number of actions.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  model = NatureDQNNetwork(num_actions, network_type)
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
  model = RainbowNetwork(num_actions, num_atoms, support)
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
  model = ImplicitQuantileNetwork(num_actions, quantile_embedding_dim)
  net = model(state, num_quantiles)
  return network_type(
      quantile_values=net.quantile_values, quantiles=net.quantiles
  )


### Generic Gym networks ###
@gin.configurable
class BasicDiscreteDomainNetwork(tf.keras.layers.Layer):
  """The fully connected network used to compute the agent's Q-values.

  This sub network used within various other models. Since it is an inner
  block, we define it as a layer. These sub networks normalize their inputs to
  lie in range [-1, 1], using min_/max_vals. It supports both DQN- and
  Rainbow- style networks.
  Attributes:
    min_vals: float, minimum attainable values (must be same shape as `state`).
    max_vals: float, maximum attainable values (must be same shape as `state`).
    num_actions: int, number of actions.
    num_atoms: int or None, if None will construct a DQN-style network,
      otherwise will construct a Rainbow-style network.
    name: str, used to create scope for network parameters.
    activation_fn: function, passed to the layer constructors.
  """

  def __init__(
      self,
      min_vals,
      max_vals,
      num_actions,
      num_atoms=None,
      name=None,
      activation_fn=tf.keras.activations.relu,
  ):
    super(BasicDiscreteDomainNetwork, self).__init__(name=name)
    self.num_actions = num_actions
    self.num_atoms = num_atoms
    self.min_vals = min_vals
    self.max_vals = max_vals
    # Defining layers.
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(
        512, activation=activation_fn, name='fully_connected'
    )
    self.dense2 = tf.keras.layers.Dense(
        512, activation=activation_fn, name='fully_connected'
    )
    if num_atoms is None:
      self.last_layer = tf.keras.layers.Dense(
          num_actions, name='fully_connected'
      )
    else:
      self.last_layer = tf.keras.layers.Dense(
          num_actions * num_atoms, name='fully_connected'
      )

  def call(self, state):
    """Creates the output tensor/op given the state tensor as input."""
    x = tf.cast(state, tf.float32)
    x = self.flatten(x)
    if self.min_vals is not None:
      x -= self.min_vals
      x /= self.max_vals - self.min_vals
      x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    x = self.dense1(x)
    x = self.dense2(x)
    x = self.last_layer(x)
    return x


@gin.configurable
class CartpoleDQNNetwork(tf.keras.Model):
  """Keras DQN network for Cartpole."""

  def __init__(self, num_actions, name=None):
    """Builds the deep network used to compute the agent's Q-values.

    It rescales the input features so they lie in range [-1, 1].

    Args:
      num_actions: int, number of actions.
      name: str, used to create scope for network parameters.
    """
    super(CartpoleDQNNetwork, self).__init__(name=name)
    self.net = BasicDiscreteDomainNetwork(
        gym_lib.CARTPOLE_MIN_VALS, gym_lib.CARTPOLE_MAX_VALS, num_actions
    )

  def call(self, state):
    """Creates the output tensor/op given the state tensor as input."""
    x = self.net(state)
    return DQNNetworkType(x)


class FourierBasis(object):
  """Fourier Basis linear function approximation.

  Requires the ranges for each dimension, and is thus able to use only sine or
  cosine (and uses cosine). So, this has half the coefficients that a full
  Fourier approximation would use.

  Many thanks to Will Dabney (wdabney@) for this implementation.

  From the paper:
  G.D. Konidaris, S. Osentoski and P.S. Thomas. (2011)
  Value Function Approximation in Reinforcement Learning using the Fourier Basis
  """

  def __init__(self, nvars, min_vals=0, max_vals=None, order=3):
    self.order = order
    self.min_vals = min_vals
    self.max_vals = max_vals
    terms = itertools.product(range(order + 1), repeat=nvars)

    # Removing first iterate because it corresponds to the constant bias
    self.multipliers = tf.constant(
        [list(map(int, x)) for x in terms][1:], dtype=tf.float32
    )

  def scale(self, values):
    shifted = values - self.min_vals
    if self.max_vals is None:
      return shifted

    return shifted / (self.max_vals - self.min_vals)

  def compute_features(self, features):
    # Important to rescale features to be between [0,1]
    scaled = self.scale(features)
    return tf.cos(np.pi * tf.matmul(scaled, self.multipliers, transpose_b=True))


@gin.configurable
class FourierDQNNetwork(tf.keras.Model):
  """Keras model for DQN."""

  def __init__(
      self, min_vals, max_vals, num_actions, fourier_basis_order=3, name=None
  ):
    """Builds the function approximator used to compute the agent's Q-values.

    It uses the features of the FourierBasis class and a linear layer
    without bias.

    Value Function Approximation in Reinforcement Learning using the Fourier
    Basis", Konidaris, Osentoski and Thomas (2011).

    Args:
      min_vals: float, minimum attainable values (must be same shape as
        `state`).
      max_vals: float, maximum attainable values (must be same shape as
        `state`).
      num_actions: int, number of actions.
      fourier_basis_order: int, order of the Fourier basis functions.
      name: str, used to create scope for network parameters.
    """
    super(FourierDQNNetwork, self).__init__(name=name)
    self.num_actions = num_actions
    self.fourier_basis_order = fourier_basis_order
    self.min_vals = min_vals
    self.max_vals = max_vals
    # Defining layers.
    self.flatten = tf.keras.layers.Flatten()
    self.last_layer = tf.keras.layers.Dense(
        num_actions, use_bias=False, name='fully_connected'
    )

  def call(self, state):
    """Creates the output tensor/op given the state tensor as input."""
    x = tf.cast(state, tf.float32)
    x = self.flatten(x)
    # Since FourierBasis needs the shape of the input, we can only initialize
    # it during the first forward pass when we know the shape of the input.
    if not hasattr(self, 'feature_generator'):
      self.feature_generator = FourierBasis(
          x.get_shape().as_list()[-1],
          self.min_vals,
          self.max_vals,
          order=self.fourier_basis_order,
      )
    x = self.feature_generator.compute_features(x)
    x = self.last_layer(x)
    return DQNNetworkType(x)


@gin.configurable
class CartpoleFourierDQNNetwork(FourierDQNNetwork):
  """Keras network for fourier Cartpole."""

  def __init__(self, num_actions, name=None):
    """Builds the function approximator used to compute the agent's Q-values.

    It uses the Fourier basis features and a linear function approximator.

    Args:
      num_actions: int, number of actions.
      name: str, used to create scope for network parameters.
    """
    super(CartpoleFourierDQNNetwork, self).__init__(
        gym_lib.CARTPOLE_MIN_VALS,
        gym_lib.CARTPOLE_MAX_VALS,
        num_actions,
        name=name,
    )


@gin.configurable
class CartpoleRainbowNetwork(tf.keras.Model):
  """Keras Rainbow network for Cartpole."""

  def __init__(self, num_actions, num_atoms, support, name=None):
    """Builds the deep network used to compute the agent's Q-values.

    It rescales the input features to a range that yields improved performance.

    Args:
      num_actions: int, number of actions.
      num_atoms: int, the number of buckets of the value function distribution.
      support: tf.linspace, the support of the Q-value distribution.
      name: str, used to create scope for network parameters.
    """
    super(CartpoleRainbowNetwork, self).__init__(name=name)
    self.net = BasicDiscreteDomainNetwork(
        gym_lib.CARTPOLE_MIN_VALS,
        gym_lib.CARTPOLE_MAX_VALS,
        num_actions,
        num_atoms=num_atoms,
    )
    self.num_actions = num_actions
    self.num_atoms = num_atoms
    self.support = support

  def call(self, state):
    x = self.net(state)
    logits = tf.reshape(x, [-1, self.num_actions, self.num_atoms])
    probabilities = tf.keras.activations.softmax(logits)
    q_values = tf.reduce_sum(self.support * probabilities, axis=2)
    return RainbowNetworkType(q_values, logits, probabilities)


@gin.configurable
class AcrobotDQNNetwork(tf.keras.Model):
  """Keras DQN network for Acrobot."""

  def __init__(self, num_actions, name=None):
    """Builds the deep network used to compute the agent's Q-values.

    It rescales the input features to a range that yields improved performance.

    Args:
      num_actions: int, number of actions.
      name: str, used to create scope for network parameters.
    """
    super(AcrobotDQNNetwork, self).__init__(name=name)
    self.net = BasicDiscreteDomainNetwork(
        gym_lib.ACROBOT_MIN_VALS, gym_lib.ACROBOT_MAX_VALS, num_actions
    )

  def call(self, state):
    x = self.net(state)
    return DQNNetworkType(x)


@gin.configurable
class AcrobotFourierDQNNetwork(FourierDQNNetwork):
  """Keras fourier DQN network for Acrobot."""

  def __init__(self, num_actions, name=None):
    """Builds the function approximator used to compute the agent's Q-values.

    It uses the Fourier basis features and a linear function approximator.

    Args:
      num_actions: int, number of actions.
      name: str, used to create scope for network parameters.
    """
    super(AcrobotFourierDQNNetwork, self).__init__(
        gym_lib.ACROBOT_MIN_VALS,
        gym_lib.ACROBOT_MAX_VALS,
        num_actions,
        name=name,
    )


@gin.configurable
class AcrobotRainbowNetwork(tf.keras.Model):
  """Keras Rainbow network for Acrobot."""

  def __init__(self, num_actions, num_atoms, support, name=None):
    """Builds the deep network used to compute the agent's Q-values.

    It rescales the input features to a range that yields improved performance.

    Args:
      num_actions: int, number of actions.
      num_atoms: int, the number of buckets of the value function distribution.
      support: Tensor, the support of the Q-value distribution.
      name: str, used to create scope for network parameters.
    """
    super(AcrobotRainbowNetwork, self).__init__(name=name)
    self.net = BasicDiscreteDomainNetwork(
        gym_lib.ACROBOT_MIN_VALS,
        gym_lib.ACROBOT_MAX_VALS,
        num_actions,
        num_atoms=num_atoms,
    )
    self.num_actions = num_actions
    self.num_atoms = num_atoms
    self.support = support

  def call(self, state):
    x = self.net(state)
    logits = tf.reshape(x, [-1, self.num_actions, self.num_atoms])
    probabilities = tf.keras.activations.softmax(logits)
    q_values = tf.reduce_sum(self.support * probabilities, axis=2)
    return RainbowNetworkType(q_values, logits, probabilities)


@gin.configurable
class LunarLanderDQNNetwork(tf.keras.Model):
  """Keras DQN network for LunarLander."""

  def __init__(self, num_actions, name=None):
    """Builds the deep network used to compute the agent's Q-values.

    Args:
      num_actions: int, number of actions.
      name: str, used to create scope for network parameters.
    """
    super(LunarLanderDQNNetwork, self).__init__(name=name)
    self.net = BasicDiscreteDomainNetwork(None, None, num_actions)

  def call(self, state):
    """Creates the output tensor/op given the state tensor as input."""
    x = self.net(state)
    return DQNNetworkType(x)


@gin.configurable
class MountainCarDQNNetwork(tf.keras.Model):
  """Keras DQN network for MountainCar."""

  def __init__(self, num_actions, name=None):
    """Builds the deep network used to compute the agent's Q-values.

    Args:
      num_actions: int, number of actions.
      name: str, used to create scope for network parameters.
    """
    super(MountainCarDQNNetwork, self).__init__(name=name)
    self.net = BasicDiscreteDomainNetwork(
        gym_lib.MOUNTAINCAR_MIN_VALS, gym_lib.MOUNTAINCAR_MAX_VALS, num_actions
    )

  def call(self, state):
    """Creates the output tensor/op given the state tensor as input."""
    x = self.net(state)
    return DQNNetworkType(x)


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
  layer = BasicDiscreteDomainNetwork(min_vals, max_vals, num_actions, num_atoms)
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
  model = CartpoleDQNNetwork(num_actions)
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
  model = FourierDQNNetwork(
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
  model = CartpoleFourierDQNNetwork(num_actions)
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
  model = CartpoleRainbowNetwork(num_actions, num_atoms, support)
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
  model = AcrobotDQNNetwork(num_actions)
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
  model = AcrobotFourierDQNNetwork(num_actions)
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
  model = AcrobotRainbowNetwork(num_actions, num_atoms, support)
  net = model(state)
  return network_type(net.q_values, net.logits, net.probabilities)

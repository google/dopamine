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
"""Gym-specific (non-Atari) utilities.

Some network specifications specific to certain Gym environments are provided
here.

Includes a wrapper class around Gym environments. This class makes general Gym
environments conformant with the API Dopamine is expecting.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math



from dopamine.discrete_domains import atari_lib
import gin
import gym
from gym.wrappers.time_limit import TimeLimit
import numpy as np
import tensorflow as tf


CARTPOLE_MIN_VALS = np.array([-2.4, -5., -math.pi/12., -math.pi*2.])
CARTPOLE_MAX_VALS = np.array([2.4, 5., math.pi/12., math.pi*2.])
ACROBOT_MIN_VALS = np.array([-1., -1., -1., -1., -5., -5.])
ACROBOT_MAX_VALS = np.array([1., 1., 1., 1., 5., 5.])
MOUNTAINCAR_MIN_VALS = np.array([-1.2, -0.07])
MOUNTAINCAR_MAX_VALS = np.array([0.6, 0.07])
gin.constant('gym_lib.CARTPOLE_OBSERVATION_SHAPE', (4, 1))
gin.constant('gym_lib.CARTPOLE_OBSERVATION_DTYPE', tf.float64)
gin.constant('gym_lib.CARTPOLE_STACK_SIZE', 1)
gin.constant('gym_lib.ACROBOT_OBSERVATION_SHAPE', (6, 1))
gin.constant('gym_lib.ACROBOT_OBSERVATION_DTYPE', tf.float64)
gin.constant('gym_lib.ACROBOT_STACK_SIZE', 1)
gin.constant('gym_lib.LUNAR_OBSERVATION_SHAPE', (8, 1))
gin.constant('gym_lib.LUNAR_OBSERVATION_DTYPE', tf.float64)
gin.constant('gym_lib.LUNAR_STACK_SIZE', 1)
gin.constant('gym_lib.MOUNTAINCAR_OBSERVATION_SHAPE', (2, 1))
gin.constant('gym_lib.MOUNTAINCAR_OBSERVATION_DTYPE', tf.float64)
gin.constant('gym_lib.MOUNTAINCAR_STACK_SIZE', 1)


MUJOCO_GAMES = ('Ant', 'HalfCheetah', 'Hopper', 'Humanoid', 'Walker2d')


@gin.configurable
def create_gym_environment(environment_name=None, version='v0'):
  """Wraps a Gym environment with some basic preprocessing.

  Args:
    environment_name: str, the name of the environment to run.
    version: str, version of the environment to run.

  Returns:
    A Gym environment with some standard preprocessing.
  """
  assert environment_name is not None


  full_game_name = '{}-{}'.format(environment_name, version)
  env = gym.make(full_game_name)
  # Strip out the TimeLimit wrapper from Gym, which caps us at 200 steps.
  if isinstance(env, TimeLimit):
    env = env.env
  # Wrap the returned environment in a class which conforms to the API expected
  # by Dopamine.
  env = GymPreprocessing(env)
  return env


@gin.configurable
class BasicDiscreteDomainNetwork(tf.keras.layers.Layer):
  """The fully connected network used to compute the agent's Q-values.

    This sub network used within various other models. Since it is an inner
    block, we define it as a layer. These sub networks normalize their inputs to
    lie in range [-1, 1], using min_/max_vals. It supports both DQN- and
    Rainbow- style networks.
    Attributes:
      min_vals: float, minimum attainable values (must be same shape as
        `state`).
      max_vals: float, maximum attainable values (must be same shape as
        `state`).
      num_actions: int, number of actions.
      num_atoms: int or None, if None will construct a DQN-style network,
        otherwise will construct a Rainbow-style network.
      name: str, used to create scope for network parameters.
      activation_fn: function, passed to the layer constructors.
  """

  def __init__(self, min_vals, max_vals, num_actions,
               num_atoms=None, name=None,
               activation_fn=tf.keras.activations.relu):
    super(BasicDiscreteDomainNetwork, self).__init__(name=name)
    self.num_actions = num_actions
    self.num_atoms = num_atoms
    self.min_vals = min_vals
    self.max_vals = max_vals
    # Defining layers.
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(512, activation=activation_fn,
                                        name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(512, activation=activation_fn,
                                        name='fully_connected')
    if num_atoms is None:
      self.last_layer = tf.keras.layers.Dense(num_actions,
                                              name='fully_connected')
    else:
      self.last_layer = tf.keras.layers.Dense(num_actions * num_atoms,
                                              name='fully_connected')

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
        CARTPOLE_MIN_VALS, CARTPOLE_MAX_VALS, num_actions)

  def call(self, state):
    """Creates the output tensor/op given the state tensor as input."""
    x = self.net(state)
    return atari_lib.DQNNetworkType(x)


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
        [list(map(int, x)) for x in terms][1:], dtype=tf.float32)

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

  def __init__(self, min_vals, max_vals, num_actions, fourier_basis_order=3,
               name=None):
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
    self.last_layer = tf.keras.layers.Dense(num_actions, use_bias=False,
                                            name='fully_connected')

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
          order=self.fourier_basis_order)
    x = self.feature_generator.compute_features(x)
    x = self.last_layer(x)
    return atari_lib.DQNNetworkType(x)


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
        CARTPOLE_MIN_VALS, CARTPOLE_MAX_VALS, num_actions, name=name)


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
        CARTPOLE_MIN_VALS, CARTPOLE_MAX_VALS, num_actions, num_atoms=num_atoms)
    self.num_actions = num_actions
    self.num_atoms = num_atoms
    self.support = support

  def call(self, state):
    x = self.net(state)
    logits = tf.reshape(x, [-1, self.num_actions, self.num_atoms])
    probabilities = tf.keras.activations.softmax(logits)
    q_values = tf.reduce_sum(self.support * probabilities, axis=2)
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)


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
        ACROBOT_MIN_VALS, ACROBOT_MAX_VALS, num_actions)

  def call(self, state):
    x = self.net(state)
    return atari_lib.DQNNetworkType(x)


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
        ACROBOT_MIN_VALS, ACROBOT_MAX_VALS, num_actions, name=name)


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
        ACROBOT_MIN_VALS, ACROBOT_MAX_VALS, num_actions, num_atoms=num_atoms)
    self.num_actions = num_actions
    self.num_atoms = num_atoms
    self.support = support

  def call(self, state):
    x = self.net(state)
    logits = tf.reshape(x, [-1, self.num_actions, self.num_atoms])
    probabilities = tf.keras.activations.softmax(logits)
    q_values = tf.reduce_sum(self.support * probabilities, axis=2)
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)


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
    return atari_lib.DQNNetworkType(x)


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
        MOUNTAINCAR_MIN_VALS, MOUNTAINCAR_MAX_VALS, num_actions)

  def call(self, state):
    """Creates the output tensor/op given the state tensor as input."""
    x = self.net(state)
    return atari_lib.DQNNetworkType(x)


@gin.configurable
class GymPreprocessing(object):
  """A Wrapper class around Gym environments."""

  def __init__(self, environment):
    self.environment = environment
    self.game_over = False

  @property
  def observation_space(self):
    return self.environment.observation_space

  @property
  def action_space(self):
    return self.environment.action_space

  @property
  def reward_range(self):
    return self.environment.reward_range

  @property
  def metadata(self):
    return self.environment.metadata

  def reset(self):
    return self.environment.reset()

  def step(self, action):
    observation, reward, game_over, info = self.environment.step(action)
    was_truncated = info.get('TimeLimit.truncated', False)
    game_over = game_over and not was_truncated
    self.game_over = game_over
    return observation, reward, game_over, info

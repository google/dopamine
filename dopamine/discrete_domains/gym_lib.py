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

import math



import gym
import numpy as np
import tensorflow as tf

import gin.tf


gin.constant('gym_lib.CARTPOLE_OBSERVATION_SHAPE', (4, 1))
gin.constant('gym_lib.CARTPOLE_OBSERVATION_DTYPE', tf.float32)
gin.constant('gym_lib.CARTPOLE_STACK_SIZE', 1)
gin.constant('gym_lib.ACROBOT_OBSERVATION_SHAPE', (6, 1))
gin.constant('gym_lib.ACROBOT_OBSERVATION_DTYPE', tf.float32)
gin.constant('gym_lib.ACROBOT_STACK_SIZE', 1)

slim = tf.contrib.slim


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
  env = env.env
  # Wrap the returned environment in a class which conforms to the API expected
  # by Dopamine.
  env = GymPreprocessing(env)
  return env


@gin.configurable
def _basic_discrete_domain_network(min_vals, max_vals, num_actions, state):
  net = tf.cast(state, tf.float32)
  net = slim.flatten(net)
  net -= min_vals
  net /= max_vals - min_vals
  net = 2.0 * net - 1.0  # Rescale in range [-1, 1].
  net = slim.fully_connected(net, 512)
  net = slim.fully_connected(net, 512)
  return slim.fully_connected(net, num_actions, activation_fn=None)


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
  min_vals = np.array([-2.4, -5, -math.pi/12, -math.pi*2])
  max_vals = np.array([2.4, 5, math.pi/12, math.pi*2])
  q_values = _basic_discrete_domain_network(min_vals, max_vals, num_actions,
                                            state)
  return network_type(q_values)


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
  min_vals = np.array([-1., -1., -1., -1., -5., -5.])
  max_vals = np.array([1., 1., 1., 1., 5., 5.])
  q_values = _basic_discrete_domain_network(min_vals, max_vals, num_actions,
                                            state)
  return network_type(q_values)


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
    self.game_over = game_over
    return observation, reward, game_over, info

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

import math

import gin
import gym as legacy_gym
from gym.wrappers.time_limit import TimeLimit
import gymnasium as gym
import numpy as np
import tensorflow as tf



CARTPOLE_MIN_VALS = np.array([-2.4, -5.0, -math.pi / 12.0, -math.pi * 2.0])
CARTPOLE_MAX_VALS = np.array([2.4, 5.0, math.pi / 12.0, math.pi * 2.0])
ACROBOT_MIN_VALS = np.array([-1.0, -1.0, -1.0, -1.0, -5.0, -5.0])
ACROBOT_MAX_VALS = np.array([1.0, 1.0, 1.0, 1.0, 5.0, 5.0])
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
def create_gym_environment(
    environment_name=None,
    version='v0',
    use_legacy_gym=False,
    use_ppo_preprocessing=False,
):
  """Wraps a Gym environment with some basic preprocessing.

  Args:
    environment_name: str, the name of the environment to run.
    version: str, version of the environment to run.
    use_legacy_gym: bool, whether to use the legacy Gym API.
    use_ppo_preprocessing: bool, whether to use PPO-specific preprocessing.

  Returns:
    A Gym environment with some standard preprocessing.
  """
  assert environment_name is not None


  full_game_name = '{}-{}'.format(environment_name, version)
  if use_legacy_gym:
    env = legacy_gym.make(full_game_name)
    if use_ppo_preprocessing:
      env = legacy_gym.wrappers.ClipAction(env)
      env = legacy_gym.wrappers.NormalizeObservation(env)
      env = legacy_gym.wrappers.TransformObservation(
          env, lambda obs: np.clip(obs, -10, 10)
      )
      env = legacy_gym.wrappers.NormalizeReward(env)
      env = legacy_gym.wrappers.TransformReward(
          env, lambda reward: np.clip(reward, -10, 10)
      )
  else:
    env = gym.make(full_game_name)
  # Strip out the TimeLimit wrapper from Gym, which caps us at 200 steps.
  if isinstance(env, TimeLimit):
    env = env.env
  # Wrap the returned environment in a class which conforms to the API expected
  # by Dopamine.
  env = GymPreprocessing(env, use_legacy_gym=use_legacy_gym)
  return env


@gin.configurable
class GymPreprocessing(object):
  """A Wrapper class around Gym environments."""

  def __init__(self, environment, use_legacy_gym=False):
    self.environment = environment
    self._use_legacy_gym = use_legacy_gym
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
    if self._use_legacy_gym:
      return self.environment.reset()

    obs, _ = self.environment.reset()
    return obs

  def step(self, action):
    if self._use_legacy_gym:
      observation, reward, game_over, info = self.environment.step(action)
      truncated = info.get('TimeLimit.truncated', False)
    else:
      observation, reward, game_over, truncated, info = self.environment.step(
          action
      )
    self.game_over = game_over and not truncated
    return observation, reward, self.game_over, info

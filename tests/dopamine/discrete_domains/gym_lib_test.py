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
"""Tests for dopamine.discrete_domains.gym_lib."""

from absl.testing import absltest
from absl.testing import parameterized
from dopamine.discrete_domains import gym_lib


class MockGymEnvironment(object):
  """Mock environment for testing with Gym/Gymnasium."""

  def __init__(self, legacy_gym_api):
    self._legacy_gym_api = legacy_gym_api
    self.observation_space = 'observation_space'
    self.action_space = 'action_space'
    self.reward_range = 'reward_range'
    self.metadata = 'metadata'

  def reset(self):
    if self._legacy_gym_api:
      return 'reset'
    return 'reset', 'info'

  def step(self, unused_action):
    if self._legacy_gym_api:
      return 'obs', 'rew', False, {}
    return 'obs', 'rew', False, False, {}


class GymPreprocessingTest(parameterized.TestCase):

  @parameterized.parameters(True, False)
  def testAll(self, use_legacy_gym):
    env = gym_lib.GymPreprocessing(
        MockGymEnvironment(use_legacy_gym), use_legacy_gym=use_legacy_gym
    )
    self.assertEqual('observation_space', env.observation_space)
    self.assertEqual('action_space', env.action_space)
    self.assertEqual('reward_range', env.reward_range)
    self.assertEqual('metadata', env.metadata)
    self.assertEqual('reset', env.reset())
    self.assertCountEqual(['obs', 'rew', False, {}], env.step(0))


if __name__ == '__main__':
  absltest.main()

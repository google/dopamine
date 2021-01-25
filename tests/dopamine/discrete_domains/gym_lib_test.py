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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from dopamine.discrete_domains import gym_lib
import tensorflow as tf


class MockGymEnvironment(object):
  """Mock environment for testing."""

  def __init__(self):
    self.observation_space = 'observation_space'
    self.action_space = 'action_space'
    self.reward_range = 'reward_range'
    self.metadata = 'metadata'

  def reset(self):
    return 'reset'

  def step(self, unused_action):
    return 'obs', 'rew', False, {}


class GymPreprocessingTest(tf.test.TestCase):

  def testAll(self):
    env = gym_lib.GymPreprocessing(MockGymEnvironment())
    self.assertEqual('observation_space', env.observation_space)
    self.assertEqual('action_space', env.action_space)
    self.assertEqual('reward_range', env.reward_range)
    self.assertEqual('metadata', env.metadata)
    self.assertEqual('reset', env.reset())
    self.assertAllEqual(['obs', 'rew', False, {}], env.step(0))


if __name__ == '__main__':
  tf.compat.v1.disable_v2_behavior()
  tf.test.main()

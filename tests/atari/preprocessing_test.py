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
"""Tests for dopamine.atari.run_experiment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from absl import flags
from dopamine.atari import preprocessing
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS


class MockALE(object):
  """Mock internal ALE for testing."""

  def __init__(self):
    pass

  def lives(self):
    return 1

  def getScreenGrayscale(self, screen):  # pylint: disable=invalid-name
    screen.fill(self.screen_value)


class MockEnvironment(object):
  """Mock environment for testing."""

  def __init__(self, screen_size=10, max_steps=10):
    self.max_steps = max_steps
    self.screen_size = screen_size
    self.ale = MockALE()
    self.observation_space = np.empty((screen_size, screen_size))
    self.game_over = False

  def reset(self):
    self.ale.screen_value = 10
    self.num_steps = 0
    return self.get_observation()

  def get_observation(self):
    observation = np.empty((self.screen_size, self.screen_size))
    return self.ale.getScreenGrayscale(observation)

  def step(self, action):
    reward = -1. if action > 0 else 1.
    self.num_steps += 1
    is_terminal = self.num_steps >= self.max_steps

    unused = 0
    self.ale.screen_value -= 2
    return (self.get_observation(), reward, is_terminal, unused)

  def render(self, mode):
    pass


class AtariPreprocessingTest(tf.test.TestCase):

  def testResetPassesObservation(self):
    env = MockEnvironment()
    env = preprocessing.AtariPreprocessing(env, frame_skip=1, screen_size=16)
    observation = env.reset()

    self.assertEqual(observation.shape, (16, 16, 1))

  def testTerminalPassedThrough(self):
    max_steps = 10
    env = MockEnvironment(max_steps=max_steps)
    env = preprocessing.AtariPreprocessing(env, frame_skip=1)
    env.reset()

    # Make sure we get the right number of steps.
    for _ in range(max_steps - 1):
      _, _, is_terminal, _ = env.step(0)
      self.assertFalse(is_terminal)

    _, _, is_terminal, _ = env.step(0)
    self.assertTrue(is_terminal)

  def testFrameSkipAccumulatesReward(self):
    frame_skip = 2
    env = MockEnvironment()
    env = preprocessing.AtariPreprocessing(env, frame_skip=frame_skip)
    env.reset()

    # Make sure we get the right number of steps. Reward is 1 when we
    # pass in action 0.
    _, reward, _, _ = env.step(0)
    self.assertEqual(reward, frame_skip)

  def testMaxFramePooling(self):
    frame_skip = 2
    env = MockEnvironment()
    env = preprocessing.AtariPreprocessing(env, frame_skip=frame_skip)
    env.reset()

    # The first observation is 2, the second 0; max is 2.
    observation, _, _, _ = env.step(0)
    self.assertTrue((observation == 8).all())

if __name__ == '__main__':
  tf.test.main()

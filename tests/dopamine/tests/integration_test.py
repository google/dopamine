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
"""End to end integration tests for Dopamine package."""

import datetime
import os
import shutil



from absl import flags
from dopamine.discrete_domains import train
import tensorflow as tf

import gin.tf


FLAGS = flags.FLAGS


class AtariIntegrationTest(tf.test.TestCase):
  """Tests for Atari environment with various agents.

  """

  def setUp(self):
    FLAGS.base_dir = os.path.join(
        '/tmp/dopamine_tests',
        datetime.datetime.utcnow().strftime('run_%Y_%m_%d_%H_%M_%S'))
    self._checkpoint_dir = os.path.join(FLAGS.base_dir, 'checkpoints')
    self._logging_dir = os.path.join(FLAGS.base_dir, 'logs')
    FLAGS.alsologtostderr = True
    gin.clear_config()

  def quickDqnFlags(self):
    """Assign flags for a quick run of DQNAgent."""
    FLAGS.gin_files = ['dopamine/agents/dqn/configs/dqn.gin']
    FLAGS.gin_bindings = [
        'Runner.training_steps=100',
        'Runner.evaluation_steps=10',
        'Runner.num_iterations=1',
        'Runner.max_steps_per_episode=100',
        'dqn_agent.DQNAgent.min_replay_history=500',
        'WrappedReplayBuffer.replay_capacity=100'
    ]

  def quickRainbowFlags(self):
    """Assign flags for a quick run of RainbowAgent."""
    FLAGS.gin_files = [
        'dopamine/agents/rainbow/configs/rainbow.gin'
    ]
    FLAGS.gin_bindings = [
        'Runner.training_steps=100',
        'Runner.evaluation_steps=10',
        'Runner.num_iterations=1',
        'Runner.max_steps_per_episode=100',
        "rainbow_agent.RainbowAgent.replay_scheme='prioritized'",
        'rainbow_agent.RainbowAgent.min_replay_history=500',
        'WrappedReplayBuffer.replay_capacity=100'
    ]

  def verifyFilesCreated(self, base_dir):
    """Verify that files have been created."""
    # Check checkpoint files
    self.assertTrue(
        os.path.exists(os.path.join(self._checkpoint_dir, 'ckpt.0')))
    self.assertTrue(
        os.path.exists(os.path.join(self._checkpoint_dir, 'checkpoint')))
    self.assertTrue(
        os.path.exists(
            os.path.join(self._checkpoint_dir,
                         'sentinel_checkpoint_complete.0')))
    # Check log files
    self.assertTrue(os.path.exists(os.path.join(self._logging_dir, 'log_0')))

  def testIntegrationDqn(self):
    """Test the DQN agent."""
    tf.logging.info('####### Training the DQN agent #####')
    tf.logging.info('####### DQN base_dir: {}'.format(FLAGS.base_dir))
    self.quickDqnFlags()
    train.main([])
    self.verifyFilesCreated(FLAGS.base_dir)
    shutil.rmtree(FLAGS.base_dir)

  def testIntegrationRainbow(self):
    """Test the rainbow agent."""
    tf.logging.info('####### Training the Rainbow agent #####')
    tf.logging.info('####### Rainbow base_dir: {}'.format(FLAGS.base_dir))
    self.quickRainbowFlags()
    train.main([])
    self.verifyFilesCreated(FLAGS.base_dir)
    shutil.rmtree(FLAGS.base_dir)


if __name__ == '__main__':
  tf.test.main()

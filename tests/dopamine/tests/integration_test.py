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
from absl import logging

from dopamine.discrete_domains import train
import gin.tf
import tensorflow as tf


FLAGS = flags.FLAGS


class AtariIntegrationTest(tf.test.TestCase):
  """Tests for Atari environment with various agents.

  """

  def setUp(self):
    super(AtariIntegrationTest, self).setUp()
    FLAGS.base_dir = os.path.join(
        '/tmp/dopamine_tests',
        datetime.datetime.utcnow().strftime('run_%Y_%m_%d_%H_%M_%S'))
    self._checkpoint_dir = os.path.join(FLAGS.base_dir, 'checkpoints')
    self._logging_dir = os.path.join(FLAGS.base_dir, 'logs')
    FLAGS.alsologtostderr = True
    gin.clear_config()

  def quick_dqn_flags(self):
    """Assign flags for a quick run of DQNAgent."""
    FLAGS.gin_files = ['dopamine/jax/agents/dqn/configs/dqn.gin']
    FLAGS.gin_bindings = [
        'Runner.training_steps=100',
        'Runner.evaluation_steps=10',
        'Runner.num_iterations=1',
        'Runner.max_steps_per_episode=100',
        'dqn_agent.JaxDQNAgent.min_replay_history=500',
        'WrappedReplayBuffer.replay_capacity=100'
    ]

  def quick_rainbow_flags(self):
    """Assign flags for a quick run of RainbowAgent."""
    FLAGS.gin_files = [
        'dopamine/jax/agents/rainbow/configs/rainbow.gin'
    ]
    FLAGS.gin_bindings = [
        'Runner.training_steps=100',
        'Runner.evaluation_steps=10',
        'Runner.num_iterations=1',
        'Runner.max_steps_per_episode=100',
        "rainbow_agent.JaxRainbowAgent.replay_scheme='prioritized'",
        'rainbow_agent.JaxRainbowAgent.min_replay_history=500',
        'WrappedReplayBuffer.replay_capacity=100'
    ]

  def verify_files_created(self):
    """Verify that files have been created."""
    # Check checkpoint files
    self.assertTrue(
        os.path.exists(os.path.join(self._checkpoint_dir, 'ckpt.0')))
    self.assertTrue(
        os.path.exists(
            os.path.join(self._checkpoint_dir,
                         'sentinel_checkpoint_complete.0')))
    # Check log files
    self.assertTrue(os.path.exists(os.path.join(self._logging_dir, 'log_0')))

  def testIntegrationDqn(self):
    """Test the DQN agent."""
    logging.info('####### Training the DQN agent #####')
    logging.info('####### DQN base_dir: %s', FLAGS.base_dir)
    self.quick_dqn_flags()
    train.main([])
    self.verify_files_created()
    shutil.rmtree(FLAGS.base_dir)

  def testIntegrationRainbow(self):
    """Test the rainbow agent."""
    logging.info('####### Training the Rainbow agent #####')
    logging.info('####### Rainbow base_dir: %s', FLAGS.base_dir)
    self.quick_rainbow_flags()
    train.main([])
    self.verify_files_created()
    shutil.rmtree(FLAGS.base_dir)


if __name__ == '__main__':
  tf.compat.v1.disable_v2_behavior()
  tf.test.main()

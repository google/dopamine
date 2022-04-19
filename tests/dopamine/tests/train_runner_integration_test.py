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
"""End to end tests for TrainRunner."""

import datetime
import os
import shutil

from absl import flags
from absl import logging

from dopamine.discrete_domains import train
import tensorflow as tf

FLAGS = flags.FLAGS


class TrainRunnerIntegrationTest(tf.test.TestCase):
  """Tests for Atari environment with various agents.

  """

  def setUp(self):
    super(TrainRunnerIntegrationTest, self).setUp()
    FLAGS.base_dir = os.path.join(
        '/tmp/dopamine_tests',
        datetime.datetime.utcnow().strftime('run_%Y_%m_%d_%H_%M_%S'))
    self._checkpoint_dir = os.path.join(FLAGS.base_dir, 'checkpoints')
    self._logging_dir = os.path.join(FLAGS.base_dir, 'logs')

  def quickDqnFlags(self):
    """Assign flags for a quick run of DQN agent."""
    FLAGS.gin_files = ['dopamine/jax/agents/dqn/configs/dqn.gin']
    FLAGS.gin_bindings = [
        "create_runner.schedule='continuous_train'",
        'Runner.training_steps=100',
        'Runner.evaluation_steps=10',
        'Runner.num_iterations=1',
        'Runner.max_steps_per_episode=100',
        'dqn_agent.JaxDQNAgent.min_replay_history=500',
        'WrappedReplayBuffer.replay_capacity=100'
    ]
    FLAGS.alsologtostderr = True

  def verifyFilesCreated(self, base_dir):
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
    self.quickDqnFlags()
    train.main([])
    self.verifyFilesCreated(FLAGS.base_dir)
    shutil.rmtree(FLAGS.base_dir)


if __name__ == '__main__':
  tf.compat.v1.disable_v2_behavior()
  tf.test.main()

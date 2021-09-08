# coding=utf-8
# Copyright 2021 The Dopamine Authors.
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
"""Tests for running Atari 100k agents."""

import os
import shutil

from absl import flags
from absl import logging

from absl.testing import absltest
from absl.testing import parameterized
from dopamine.labs.atari_100k import train
import gin

FLAGS = flags.FLAGS


def QuickAgentFlags():
  """Assigns flags for a quick run of an Atari 100k agent."""
  FLAGS.gin_bindings = [
      'Runner.training_steps=100', 'MaxEpisodeEvalRunner.num_eval_episodes=2',
      'Runner.num_iterations=1', 'Runner.max_steps_per_episode=100',
      'JaxDQNAgent.min_replay_history=100',
      'OutOfGraphPrioritizedReplayBuffer.replay_capacity=10000',
  ]
  FLAGS.alsologtostderr = True


def SetAgentConfig(agent_name='der'):
  """Sets gin configuration and name for an Atari 100k agent."""
  FLAGS.gin_files = [
      f'dopamine/labs/atari_100k/configs/{agent_name}.gin'
  ]
  FLAGS.agent = agent_name


class RunnerIntegrationTest(parameterized.TestCase):
  """Tests for Atari 100k agents.

  """

  def setUp(self):
    super().setUp()
    FLAGS.base_dir = '/tmp/dopamine_tests'
    self._checkpoint_dir = os.path.join(FLAGS.base_dir, 'checkpoints')
    self._logging_dir = os.path.join(FLAGS.base_dir, 'logs')
    gin.clear_config()

  def VerifyFilesCreated(self):
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

  @parameterized.parameters('OTRainbow', 'DER', 'DrQ', 'DrQ_eps')
  def testIntegration(self, agent_name):
    """Test the DER agent."""
    logging.info('####### Training the %s agent #####', agent_name)
    logging.info('####### %s base_dir: %s', agent_name, FLAGS.base_dir)
    QuickAgentFlags()
    SetAgentConfig(agent_name)
    train.main([])
    print(os.listdir(self._logging_dir))
    print(os.listdir(self._checkpoint_dir))
    self.VerifyFilesCreated()
    shutil.rmtree(FLAGS.base_dir)


if __name__ == '__main__':
  absltest.main()

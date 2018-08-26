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
"""Tests for dopamine.atari.train."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function




from absl import flags
from absl.testing import flagsaver
from dopamine.atari import run_experiment
from dopamine.atari import train
import mock
import tensorflow as tf

import gin.tf

FLAGS = flags.FLAGS


class TrainTest(tf.test.TestCase):

  def testCreateDQNAgent(self):
    FLAGS.agent_name = 'dqn'
    with mock.patch.object(train, 'dqn_agent') as mock_dqn_agent:

      def mock_fn(unused_sess, num_actions):
        return num_actions * 10

      mock_dqn_agent.DQNAgent.side_effect = mock_fn
      environment = mock.Mock()
      environment.action_space.n = 7
      self.assertEqual(70, train.create_agent(self.test_session(), environment))

  def testCreateRainbowAgent(self):
    FLAGS.agent_name = 'rainbow'
    with mock.patch.object(train, 'rainbow_agent') as mock_rainbow_agent:

      def mock_fn(unused_sess, num_actions):
        return num_actions * 10

      mock_rainbow_agent.RainbowAgent.side_effect = mock_fn
      environment = mock.Mock()
      environment.action_space.n = 7
      self.assertEqual(70, train.create_agent(self.test_session(), environment))

  @mock.patch.object(run_experiment, 'Runner')
  def testCreateRunnerUnknown(self, mock_runner_constructor):
    mock_create_agent = mock.Mock()
    base_dir = '/tmp'
    FLAGS.schedule = 'unknown_schedule'
    with self.assertRaisesRegexp(ValueError, 'Unknown schedule'):
      train.create_runner(base_dir, mock_create_agent)

  @mock.patch.object(run_experiment, 'Runner')
  def testCreateRunner(self, mock_runner_constructor):
    mock_create_agent = mock.Mock()
    base_dir = '/tmp'
    train.create_runner(base_dir, mock_create_agent)
    self.assertEqual(1, mock_runner_constructor.call_count)
    mock_args, _ = mock_runner_constructor.call_args
    self.assertEqual(base_dir, mock_args[0])
    self.assertEqual(mock_create_agent, mock_args[1])

  @flagsaver.flagsaver(schedule='continuous_train')
  @mock.patch.object(run_experiment, 'TrainRunner')
  def testCreateTrainRunner(self, mock_runner_constructor):
    mock_create_agent = mock.Mock()
    base_dir = '/tmp'
    train.create_runner(base_dir, mock_create_agent)
    self.assertEqual(1, mock_runner_constructor.call_count)
    mock_args, _ = mock_runner_constructor.call_args
    self.assertEqual(base_dir, mock_args[0])
    self.assertEqual(mock_create_agent, mock_args[1])

  @flagsaver.flagsaver(gin_files=['file1', 'file2', 'file3'])
  @flagsaver.flagsaver(gin_bindings=['binding1', 'binding2'])
  @mock.patch.object(gin, 'parse_config_files_and_bindings')
  @mock.patch.object(run_experiment, 'Runner')
  def testLaunchExperiment(
      self, mock_runner_constructor, mock_parse_config_files_and_bindings):
    mock_create_agent = mock.Mock()
    mock_runner = mock.Mock()
    mock_runner_constructor.return_value = mock_runner

    def mock_create_runner(unused_base_dir, unused_create_agent_fn):
      return mock_runner

    train.launch_experiment(mock_create_runner, mock_create_agent)
    self.assertEqual(1, mock_parse_config_files_and_bindings.call_count)
    mock_args, mock_kwargs = mock_parse_config_files_and_bindings.call_args
    self.assertEqual(FLAGS.gin_files, mock_args[0])
    self.assertEqual(FLAGS.gin_bindings, mock_kwargs['bindings'])
    self.assertFalse(mock_kwargs['skip_unknown'])
    self.assertEqual(1, mock_runner.run_experiment.call_count)


if __name__ == '__main__':
  tf.test.main()

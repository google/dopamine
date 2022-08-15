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
"""Tests for run_experiment."""

from typing import Type
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dopamine.continuous_domains import run_experiment
from dopamine.discrete_domains import gym_lib
from dopamine.discrete_domains import run_experiment as base_run_experiment
from dopamine.jax.agents.sac import sac_agent
import gin
from gym import spaces


class RunExperimentTest(parameterized.TestCase):

  def setUp(self):
    super(RunExperimentTest, self).setUp()

    self.env = self.enter_context(
        mock.patch.object(gym_lib, 'GymPreprocessing', autospec=True))
    self.env.observation_space = spaces.Box(0.0, 1.0, (5,))
    self.env.action_space = spaces.Box(0.0, 1.0, (4,))

    # Required for creating a SAC agent in create_agent tests.
    gin.bind_parameter(
        'circular_replay_buffer.OutOfGraphReplayBuffer.replay_capacity', 10)
    gin.bind_parameter(
        'circular_replay_buffer.OutOfGraphReplayBuffer.batch_size', 2)

    # Required for creating continuous runners.
    gin.bind_parameter('ContinuousRunner.create_environment_fn',
                       lambda: self.env)
    gin.bind_parameter('ContinuousTrainRunner.create_environment_fn',
                       lambda: self.env)

  def testCreateContinuousAgentReturnsAgent(self):
    agent = run_experiment.create_continuous_agent(self.env, 'sac')

    self.assertIsInstance(agent, sac_agent.SACAgent)

  def testCreateContinuousAgentWithInvalidNameRaisesException(self):
    with self.assertRaises(ValueError):
      run_experiment.create_continuous_agent(self.env, 'invalid_name')

  @parameterized.named_parameters(
      dict(
          testcase_name='TrainAndEval',
          schedule='continuous_train_and_eval',
          expected=run_experiment.ContinuousRunner),
      dict(
          testcase_name='Train',
          schedule='continuous_train',
          expected=run_experiment.ContinuousTrainRunner))
  def testCreateContinuousRunnerCreatesCorrectRunner(
      self, schedule: str, expected: Type[base_run_experiment.Runner]):
    gin.bind_parameter('create_continuous_agent.agent_name', 'sac')

    runner = run_experiment.create_continuous_runner(
        self.create_tempdir().full_path, schedule)

    self.assertIsInstance(runner, expected)

  def testCreateContinuousRunnerFailsWithInvalidName(self):
    with self.assertRaises(ValueError):
      run_experiment.create_continuous_runner('unused_dir', 'invalid_name')


if __name__ == '__main__':
  absltest.main()

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
"""Tests for dopamine.jax.agents.sac.sac_agent."""

import copy
import functools
import operator
import random
from typing import Any, Tuple, Union
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dopamine.jax.agents.sac import sac_agent
import flax
import gin
import jax
import numpy as np

JaxTree = Any  # There isn't a great type hint for JaxTrees

OBSERVATION_SHAPE = (8,)
IMG_OBSERVATION_SHAPE = (84, 84, 3)


def get_mock_batch(action_shape: Tuple[int, ...],
                   observation_shape: Tuple[int, ...] = OBSERVATION_SHAPE,
                   batch_size: int = 2,
                   stack_size: int = 1):
  action_dim = functools.reduce(operator.mul, action_shape, 1)
  mock_observation = np.full(
      observation_shape + (stack_size,), 1.0, dtype=np.float32)
  mock_action = np.arange(action_dim, dtype=np.float32).reshape(action_shape)
  mock_action = mock_action / np.max(mock_action)  # Squish to action limits
  mock_reward = np.asarray(1.0)
  mock_terminal = np.asarray(0.0)
  mock_indices = np.asarray(0)
  mock_sampling_prob = np.asarray(1.0)

  return (np.stack([mock_observation for _ in range(batch_size)], axis=0),
          np.stack([mock_action for _ in range(batch_size)], axis=0),
          np.stack([mock_reward for _ in range(batch_size)], axis=0),
          np.stack([mock_observation for _ in range(batch_size)], axis=0),
          np.stack([mock_action for _ in range(batch_size)], axis=0),
          np.stack([mock_reward for _ in range(batch_size)], axis=0),
          np.stack([mock_terminal for _ in range(batch_size)], axis=0),
          np.stack([mock_indices for _ in range(batch_size)], axis=0),
          np.stack([mock_sampling_prob for _ in range(batch_size)], axis=0))


def create_agent(action_shape: Union[Tuple[int, ...], int] = 4,
                 eval_mode: bool = False,
                 min_replay_history: int = 10_000,
                 update_period: int = 4,
                 seed: int = 0,
                 observation_shape: Tuple[int, ...] = OBSERVATION_SHAPE,
                 observation_dtype: np.dtype = np.float32,
                 stack_size: int = 1) -> sac_agent.SACAgent:
  return sac_agent.SACAgent(
      action_shape=action_shape,
      action_limits=(np.full(action_shape, -1.0), np.full(action_shape, 1.0)),
      observation_shape=observation_shape,
      observation_dtype=observation_dtype,
      stack_size=stack_size,
      min_replay_history=min_replay_history,
      update_period=update_period,
      eval_mode=eval_mode,
      seed=seed)


def get_agent_params(
    agent: sac_agent.SACAgent) -> Tuple[flax.core.FrozenDict, ...]:
  return agent.network_params, agent.log_alpha


class SacAgentTest(parameterized.TestCase):

  def assertAgentParametersEqual(self, agent1: sac_agent.SACAgent,
                                 agent2: sac_agent.SACAgent):
    agent1_params = get_agent_params(agent1)
    agent2_params = get_agent_params(agent2)

    agent1_params, agent1_structure = jax.tree_flatten(agent1_params)
    agent2_params, agent2_structure = jax.tree_flatten(agent2_params)

    self.assertEqual(agent1_structure, agent2_structure,
                     'Parameter structures do not match.')

    for param1, param2 in zip(agent1_params, agent2_params):
      if (param1 != param2).any():
        self.fail(f'Parameters are not equal: {param1}, {param2}')

  def assertAgentParametersNotEqual(self, agent1: sac_agent.SACAgent,
                                    agent2: sac_agent.SACAgent):
    agent1_params = get_agent_params(agent1)
    agent2_params = get_agent_params(agent2)

    agent1_params, agent1_structure = jax.tree_flatten(agent1_params)
    agent2_params, agent2_structure = jax.tree_flatten(agent2_params)

    if agent1_structure != agent2_structure:
      self.fail('Parameter structures are not comparable.')

    for param1, param2 in zip(agent1_params, agent2_params):
      if (param1 != param2).any():
        return

    # If you get here, all parameters are equal.
    self.fail(f'All parameters are equal: {agent1_params}, {agent2_params}')

  def setUp(self):
    super(SacAgentTest, self).setUp()
    gin.bind_parameter('OutOfGraphReplayBuffer.replay_capacity', 100)
    gin.bind_parameter('OutOfGraphReplayBuffer.batch_size', 2)
    random.seed(0)  # random is used for sampling experiences from buffer.

  @parameterized.named_parameters(
      dict(testcase_name='Eval', eval_mode=True),
      dict(testcase_name='NoEval', eval_mode=False))
  def testIntegerShapedActionsMatchShapes(self, eval_mode: bool):
    action_shape = 4
    agent = create_agent(action_shape=action_shape, eval_mode=eval_mode)
    observation = np.zeros(OBSERVATION_SHAPE)

    action1 = agent.begin_episode(observation)
    action2 = agent.step(reward=0.0, observation=observation)

    self.assertIsInstance(action1, np.ndarray)
    self.assertIsInstance(action2, np.ndarray)
    self.assertEqual(action1.shape, (action_shape,))
    self.assertEqual(action2.shape, (action_shape,))

  @parameterized.named_parameters(
      dict(testcase_name='Eval1', action_shape=(4,), eval_mode=True),
      dict(testcase_name='NoEval1', action_shape=(4,), eval_mode=False),
      dict(testcase_name='Eval2', action_shape=(4, 3), eval_mode=True),
      dict(testcase_name='NoEval2', action_shape=(4, 3), eval_mode=False))
  def testTupleShapedActionsMatchShapes(self, action_shape: Tuple[int, ...],
                                        eval_mode: bool):
    agent = create_agent(action_shape=action_shape, eval_mode=eval_mode)
    observation = np.zeros(OBSERVATION_SHAPE)

    action1 = agent.begin_episode(observation)
    action2 = agent.step(reward=0.0, observation=observation)

    self.assertIsInstance(action1, np.ndarray)
    self.assertIsInstance(action2, np.ndarray)
    self.assertEqual(tuple(action1.shape), action_shape)
    self.assertEqual(tuple(action2.shape), action_shape)

  @parameterized.named_parameters(
      dict(testcase_name='SingleDimensionAction', action_shape=(4,)),
      dict(testcase_name='MultiDimensionAction', action_shape=(4, 3)))
  def testAgentParametersUpdateWhenTrained(self, action_shape: Tuple[int, ...]):
    agent = create_agent(
        action_shape=action_shape,
        eval_mode=False,
        min_replay_history=0,
        update_period=1)
    observation = np.full(OBSERVATION_SHAPE, 1.0, dtype=np.float32)

    agent._replay.sample_transition_batch = mock.MagicMock(
        return_value=get_mock_batch(action_shape))
    agent._replay.add_count = 10

    agent_before_training = copy.deepcopy(agent)
    agent.begin_episode(observation)
    agent_after_one_step = copy.deepcopy(agent)
    agent.step(reward=0.0, observation=observation)

    self.assertAgentParametersNotEqual(agent_before_training,
                                       agent_after_one_step)
    self.assertAgentParametersNotEqual(agent_after_one_step, agent)

  def testAgentParametersNotUpdatedDuringEval(self):
    action_shape = (4, 3)
    agent = create_agent(
        action_shape=action_shape,
        eval_mode=True,
        min_replay_history=0,
        update_period=1)
    observation = np.full(OBSERVATION_SHAPE, 1.0, dtype=np.float32)

    agent._replay.sample_transition_batch = mock.MagicMock(
        return_value=get_mock_batch(action_shape))
    agent._replay.add_count = 10

    agent_before_training = copy.deepcopy(agent)
    agent.begin_episode(observation)
    agent_after_one_step = copy.deepcopy(agent)
    agent.step(reward=0.0, observation=observation)

    self.assertAgentParametersEqual(agent_before_training, agent_after_one_step)
    self.assertAgentParametersEqual(agent_after_one_step, agent)

  def testRestoreAgentFromBundleRestoresParameters(self):
    agent1 = create_agent(action_shape=(4, 3), seed=123)
    agent2 = create_agent(action_shape=(4, 3), seed=456)

    agent1._replay.save = mock.create_autospec(
        agent1._replay.save, spec_set=True)
    agent2._replay.load = mock.create_autospec(
        agent2._replay.load, spec_set=True)

    # Make sure the parameters are initialized differently
    self.assertAgentParametersNotEqual(agent1, agent2)

    # No data is written to file, but the directory must exist
    tempdir = self.create_tempdir()
    bundle = agent1.bundle_and_checkpoint(tempdir, 0)
    self.assertIsNotNone(bundle)  # Ensures a bundle was returned
    agent2.unbundle(tempdir, 0, bundle)

    self.assertAgentParametersEqual(agent1, agent2)

  def testAgentTrainsWithImageObservations(self):
    action_shape = (4,)
    stack_size = 4
    agent = create_agent(
        action_shape=action_shape,
        observation_shape=IMG_OBSERVATION_SHAPE,
        observation_dtype=np.uint8,
        stack_size=stack_size,
        eval_mode=False,
        min_replay_history=0,
        update_period=1)
    observation = np.full(IMG_OBSERVATION_SHAPE, 1.0, dtype=np.float32)

    agent._replay.sample_transition_batch = mock.MagicMock(
        return_value=get_mock_batch(
            action_shape=action_shape,
            observation_shape=IMG_OBSERVATION_SHAPE,
            stack_size=stack_size))
    agent._replay.add_count = 10

    agent_before_training = copy.deepcopy(agent)
    agent.begin_episode(observation)
    agent_after_one_step = copy.deepcopy(agent)
    agent.step(reward=0.0, observation=observation)

    self.assertAgentParametersNotEqual(agent_before_training,
                                       agent_after_one_step)
    self.assertAgentParametersNotEqual(agent_after_one_step, agent)


if __name__ == '__main__':
  absltest.main()

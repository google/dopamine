# coding=utf-8
# Copyright 2024 The Dopamine Authors.
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
"""Tests for dopamine.jax.agents.ppo.ppo_agent."""

from typing import Tuple, Union
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dopamine.jax.agents.ppo import ppo_agent
import flax
import gin
import jax
import jax.numpy as jnp
import numpy as np


OBSERVATION_SHAPE = (8,)


def create_agent(
    action_shape: Union[Tuple[int, ...], int] = 4,  # pytype: disable=annotation-type-mismatch  # numpy-scalars
    eval_mode: bool = False,
    update_period: int = 100,
    seed: int = 0,
    observation_shape: Tuple[int, ...] = OBSERVATION_SHAPE,
    stack_size: int = 1,
) -> ppo_agent.PPOAgent:
  return ppo_agent.PPOAgent(
      action_shape=action_shape,
      action_limits=None,
      observation_shape=observation_shape,
      stack_size=stack_size,
      update_period=update_period,
      eval_mode=eval_mode,
      seed=seed,
  )


def get_agent_params(
    agent: ppo_agent.PPOAgent,
) -> flax.core.FrozenDict:
  return agent.network_params


class PPOAgentTest(parameterized.TestCase):

  def assertAgentParametersEqual(
      self, agent1: ppo_agent.PPOAgent, agent2: ppo_agent.PPOAgent
  ):
    agent1_params = get_agent_params(agent1)
    agent2_params = get_agent_params(agent2)

    agent1_params, agent1_structure = jax.tree.flatten(agent1_params)
    agent2_params, agent2_structure = jax.tree.flatten(agent2_params)

    self.assertEqual(
        agent1_structure, agent2_structure, 'Parameter structures do not match.'
    )

    for param1, param2 in zip(agent1_params, agent2_params):
      if (param1 != param2).any():
        self.fail(f'Parameters are not equal: {param1}, {param2}')

  def assertAgentParametersNotEqual(
      self, agent1: ppo_agent.PPOAgent, agent2: ppo_agent.PPOAgent
  ):
    agent1_params = get_agent_params(agent1)
    agent2_params = get_agent_params(agent2)

    agent1_params, agent1_structure = jax.tree.flatten(agent1_params)
    agent2_params, agent2_structure = jax.tree.flatten(agent2_params)

    if agent1_structure != agent2_structure:
      self.fail('Parameter structures are not comparable.')

    for param1, param2 in zip(agent1_params, agent2_params):
      if (param1 != param2).any():
        return

    # If you get here, all parameters are equal.
    self.fail(f'All parameters are equal: {agent1_params}, {agent2_params}')

  def setUp(self):
    super(PPOAgentTest, self).setUp()
    gin.bind_parameter('ReplayBuffer.max_capacity', 100)
    gin.bind_parameter('ReplayBuffer.batch_size', 100)

  @parameterized.named_parameters(
      dict(testcase_name='Eval', eval_mode=True),
      dict(testcase_name='NoEval', eval_mode=False),
  )
  def test_integer_shaped_actions_match_shapes(self, eval_mode: bool):
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
      dict(testcase_name='NoEval2', action_shape=(4, 3), eval_mode=False),
  )
  def test_tuple_shaped_actions_match_shapes(
      self, action_shape: Tuple[int, ...], eval_mode: bool
  ):
    agent = create_agent(action_shape=action_shape, eval_mode=eval_mode)
    observation = np.zeros(OBSERVATION_SHAPE)

    action1 = agent.begin_episode(observation)
    action2 = agent.step(reward=0.0, observation=observation)

    self.assertIsInstance(action1, np.ndarray)
    self.assertIsInstance(action2, np.ndarray)
    self.assertEqual(tuple(action1.shape), action_shape)
    self.assertEqual(tuple(action2.shape), action_shape)

  def test_restore_agent_from_bundle_restores_parameters(self):
    agent1 = create_agent(action_shape=(4, 3), seed=123)
    agent2 = create_agent(action_shape=(4, 3), seed=456)

    agent1._replay.save = mock.create_autospec(
        agent1._replay.save, spec_set=True
    )
    agent2._replay.load = mock.create_autospec(
        agent2._replay.load, spec_set=True
    )

    # Make sure the parameters are initialized differently
    self.assertAgentParametersNotEqual(agent1, agent2)

    # No data is written to file, but the directory must exist
    tempdir = self.create_tempdir()
    bundle = agent1.bundle_and_checkpoint(tempdir, 0)
    self.assertIsNotNone(bundle)  # Ensures a bundle was returned
    agent2.unbundle(tempdir, 0, bundle)

    self.assertAgentParametersEqual(agent1, agent2)

  def test_calculate_advantages_and_returns_shapes(self):
    batch_shape = (10,)
    q_values = jnp.zeros(batch_shape)
    next_q_value = jnp.array(0.0)
    rewards = jnp.zeros(batch_shape)
    terminals = jnp.zeros(batch_shape)
    gamma = 0.99
    lambd = 0.95
    advantages, returns = ppo_agent.calculate_advantages_and_returns(
        q_values,
        next_q_value,
        rewards,
        terminals,
        gamma,
        lambd,
    )
    self.assertEqual(advantages.shape, batch_shape)
    self.assertEqual(advantages.dtype, jnp.float32)
    self.assertEqual(returns.shape, batch_shape)
    self.assertEqual(returns.dtype, jnp.float32)

  @parameterized.named_parameters(
      dict(
          testcase_name='infinite_time_rewards',
          discounts=np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
          gamma=1.0,
          lambda_=1.0,
      ),
      dict(
          testcase_name='immediate_rewards',
          discounts=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
          gamma=0.0,
          lambda_=0.0,
      ),
      dict(
          testcase_name='lambda_discount',
          discounts=np.array([4.900995, 3.940399, 2.9701, 1.99, 1.0]),
          gamma=1.0,
          lambda_=0.99,
      ),
      dict(
          testcase_name='gamma_discount',
          discounts=np.array([4.900995, 3.940399, 2.9701, 1.99, 1.0]),
          gamma=0.99,
          lambda_=1.0,
      ),
      dict(
          testcase_name='gamma_and_lambda_discount',
          discounts=np.array([4.80492, 3.882176, 2.940696, 1.9801, 1.0]),
          gamma=0.99,
          lambda_=0.99,
      ),
  )
  def test_calculate_advantages_and_returns_discounting(
      self, discounts, gamma, lambda_
  ):
    batch_shape = (5,)
    q_values = jnp.zeros(batch_shape)
    next_q_value = jnp.array(0.0)
    # reward is 1.0 for all timesteps
    rewards = jnp.ones(batch_shape)
    terminals = jnp.zeros(batch_shape)
    advantages, returns = ppo_agent.calculate_advantages_and_returns(
        q_values,
        next_q_value,
        rewards,
        terminals,
        gamma,
        lambda_,
    )
    np.testing.assert_array_almost_equal(returns, discounts)
    np.testing.assert_array_almost_equal(advantages, discounts)

  @parameterized.named_parameters(
      dict(
          testcase_name='full_q_values',
          q_values=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
          next_q_value=np.array(6.0),
          expected_returns=np.array([6.0, 6.0, 6.0, 6.0, 6.0]),
          expected_advantages=np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
      ),
      dict(
          testcase_name='partial_q_values',
          q_values=np.array([0.0, 0.0, 0.0, 1.0, 2.0]),
          next_q_value=np.array(3.0),
          expected_returns=np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
          expected_advantages=np.array([3.0, 3.0, 3.0, 2.0, 1.0]),
      ),
      dict(
          testcase_name='only_last_q_value',
          q_values=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
          next_q_value=np.array(1.0),
          expected_returns=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
          expected_advantages=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
      ),
      dict(
          testcase_name='zero_q_values',
          q_values=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
          next_q_value=np.array(0.0),
          expected_returns=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
          expected_advantages=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
      ),
  )
  def test_calculate_advantages_and_returns_q_value_calculations(
      self, q_values, next_q_value, expected_returns, expected_advantages
  ):
    batch_shape = (5,)
    rewards = jnp.zeros(batch_shape)
    terminals = jnp.zeros(batch_shape)
    gamma = 1.0
    lambda_ = 1.0
    advantages, returns = ppo_agent.calculate_advantages_and_returns(
        q_values,
        next_q_value,
        rewards,
        terminals,
        gamma,
        lambda_,
    )
    np.testing.assert_array_almost_equal(returns, expected_returns)
    np.testing.assert_array_almost_equal(advantages, expected_advantages)

  def test_create_minibatches_and_shuffle_shapes(self):
    action_shape = (4,)
    batch_shape = (10,)
    batch_size = 5
    states = jnp.zeros(batch_shape + OBSERVATION_SHAPE)
    actions = jnp.zeros(batch_shape + action_shape)
    returns = jnp.zeros(batch_shape)
    advantages = jnp.zeros(batch_shape)
    log_probability = jnp.zeros(batch_shape)
    q_values = jnp.zeros(batch_shape)
    key = jax.random.PRNGKey(0)
    (
        num_batches,
        states,
        actions,
        returns,
        advantages,
        log_probability,
        q_values,
    ) = ppo_agent.create_minibatches_and_shuffle(
        states,
        actions,
        returns,
        advantages,
        log_probability,
        q_values,
        batch_size,
        key,
    )
    self.assertEqual(
        states.shape, (num_batches, batch_size) + OBSERVATION_SHAPE
    )
    self.assertEqual(actions.shape, (num_batches, batch_size) + action_shape)
    self.assertEqual(returns.shape, (num_batches, batch_size))
    self.assertEqual(advantages.shape, (num_batches, batch_size))
    self.assertEqual(log_probability.shape, (num_batches, batch_size))
    self.assertEqual(q_values.shape, (num_batches, batch_size))

  def test_create_minibatches_and_shuffle_incorrect_batch_size(self):
    action_shape = (4,)
    batch_shape = (11,)
    batch_size = 5
    states = jnp.zeros(batch_shape + OBSERVATION_SHAPE)
    actions = jnp.zeros(batch_shape + action_shape)
    returns = jnp.zeros(batch_shape)
    advantages = jnp.zeros(batch_shape)
    log_probability = jnp.zeros(batch_shape)
    q_values = jnp.zeros(batch_shape)
    key = jax.random.PRNGKey(0)
    with self.assertRaises(AssertionError):
      ppo_agent.create_minibatches_and_shuffle(
          states,
          actions,
          returns,
          advantages,
          log_probability,
          q_values,
          batch_size,
          key,
      )

  @parameterized.named_parameters(
      dict(testcase_name='1dim', action_shape=(4,)),
      dict(testcase_name='2dim', action_shape=(4, 3)),
  )
  def test_select_action_shapes(self, action_shape):
    agent = create_agent(action_shape=action_shape)
    state = jnp.zeros(OBSERVATION_SHAPE)
    rng = jax.random.PRNGKey(0)
    rng, action = ppo_agent.select_action(
        agent.network_def, agent.network_params, state, rng
    )
    self.assertEqual(action.shape, action_shape)
    self.assertEqual(rng.shape, (2,))


if __name__ == '__main__':
  absltest.main()

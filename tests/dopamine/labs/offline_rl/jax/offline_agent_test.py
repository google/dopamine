# coding=utf-8
# Copyright 2023 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for dopamine.labs.offline_rl.jax agents."""

import functools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.labs.offline_rl import fixed_replay
from dopamine.labs.offline_rl.jax import networks
from dopamine.labs.offline_rl.jax import offline_dr3_agent
import gin
import jax
from jax import numpy as jnp


class OfflineAgentTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    batch_size = 8

    # Mock the replay buffer to avoid loading data over the network
    self.create_replay_buffer = functools.partial(
        mock.create_autospec, spec=fixed_replay.JaxFixedReplayBuffer,
        spec_set=True)
    observation_shape = (batch_size, *dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
                         dqn_agent.NATURE_DQN_STACK_SIZE)
    self.batch = {
        'state': jnp.full(observation_shape, 125, dtype=jnp.uint8),
        'action': jnp.full(batch_size, 0, dtype=jnp.int32),
        'reward': jnp.full(batch_size, 1.0, dtype=jnp.float32),
        'next_state': jnp.full(observation_shape, 125, dtype=jnp.uint8),
        'next_action': jnp.full(batch_size, 0, dtype=jnp.int32),
        'next_reward': jnp.full(batch_size, 1.0, dtype=jnp.float32),
        'terminal': jnp.full(batch_size, 0, dtype=jnp.uint8),
        'indices': jnp.arange(batch_size, dtype=jnp.int32)
    }

    gin.bind_parameter(
        'fixed_replay.JaxFixedReplayBuffer.replay_capacity', 100)
    gin.bind_parameter(
        'fixed_replay.JaxFixedReplayBuffer.batch_size', batch_size)
    gin.bind_parameter('JaxDQNAgent.seed', 123)  # Deterministic initialization

  def _create_agent_fn(self, agent_name):
    if agent_name == 'dr3':
      agent_fn = offline_dr3_agent.OfflineJaxDR3Agent
      gin.bind_parameter(
          'JaxDQNAgent.network', networks.JAXDQNNetworkWithRepresentations)
    else:
      raise ValueError(f'{agent_name} is a not a valid agent name.')
    return agent_fn

  def test_train_step_updates_weights(self, agent_name='dr3'):

    create_agent_fn = self._create_agent_fn(agent_name)
    agent = create_agent_fn(
        4, 'unused_string', replay_buffer_builder=self.create_replay_buffer)

    # We skip sampling from the replay buffer with a mock, and set the
    # replay elements directly to the batch we want to learn from.
    agent._sample_from_replay_buffer = mock.create_autospec(
        agent._sample_from_replay_buffer, spec_set=True)
    agent.replay_elements = self.batch

    params_before = agent.online_params
    agent.train_step()
    params_after = agent.online_params

    for i, (param1, param2) in enumerate(zip(
        jax.tree_leaves(params_before), jax.tree_leaves(params_after))):
      with self.subTest('param_set_{}'.format(i)):
        self.assertTrue((param1 != param2).any())


if __name__ == '__main__':
  absltest.main()

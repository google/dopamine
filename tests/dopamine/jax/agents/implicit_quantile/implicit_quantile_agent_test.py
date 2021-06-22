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
"""Tests for implicit quantile agent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from absl.testing import absltest
from dopamine.discrete_domains import atari_lib
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.agents.implicit_quantile import implicit_quantile_agent
from flax import linen
import gin
import jax
import jax.numpy as jnp
import numpy as onp


class ImplicitQuantileAgentTest(absltest.TestCase):

  def setUp(self):
    super(ImplicitQuantileAgentTest, self).setUp()
    self._num_actions = 4
    self.observation_shape = dqn_agent.NATURE_DQN_OBSERVATION_SHAPE
    self.observation_dtype = dqn_agent.NATURE_DQN_DTYPE
    self.stack_size = dqn_agent.NATURE_DQN_STACK_SIZE
    self.ones_state = onp.ones(
        (1,) + self.observation_shape + (self.stack_size,))
    gin.bind_parameter('OutOfGraphReplayBuffer.replay_capacity', 100)
    gin.bind_parameter('OutOfGraphReplayBuffer.batch_size', 2)

  def _create_test_agent(self):
    # This dummy network allows us to deterministically anticipate that the
    # state-action-quantile outputs will be equal to sum of the
    # corresponding quantile inputs.
    # State/Quantile shapes will be k x 1, (N x batch_size) x 1,
    # or (N' x batch_size) x 1.

    class MockImplicitQuantileNetwork(linen.Module):
      """Custom Jax model used in tests."""
      num_actions: int
      quantile_embedding_dim: int
      inputs_preprocessed: bool = False

      @linen.compact
      def __call__(self, x, num_quantiles, rng):
        del rng
        x = x.reshape((-1))  # flatten
        state_net_tiled = jnp.tile(x, [num_quantiles, 1])
        x *= state_net_tiled
        quantile_values = linen.Dense(features=self.num_actions,
                                      kernel_init=linen.initializers.ones,
                                      bias_init=linen.initializers.zeros)(x)
        quantiles = jnp.ones([num_quantiles, 1])
        return atari_lib.ImplicitQuantileNetworkType(quantile_values, quantiles)

    agent = implicit_quantile_agent.JaxImplicitQuantileAgent(
        network=MockImplicitQuantileNetwork,
        num_actions=self._num_actions,
        kappa=1.0,
        num_tau_samples=2,
        num_tau_prime_samples=3,
        num_quantile_samples=4,
        epsilon_eval=0.0)
    # This ensures non-random action choices (since epsilon_eval = 0.0) and
    # skips the train_step.
    agent.eval_mode = True
    return agent

  def testCreateAgentWithDefaults(self):
    # Verifies that we can create and train an agent with the default values.
    agent = implicit_quantile_agent.JaxImplicitQuantileAgent(num_actions=4)
    observation = onp.ones([84, 84, 1])
    agent.begin_episode(observation)
    agent.step(reward=1, observation=observation)
    agent.end_episode(reward=1)

  def testShapes(self):
    agent = self._create_test_agent()
    # Replay buffer batch size:
    self.assertEqual(agent._replay._batch_size, 2)
    for params in [agent.online_params, agent.target_network_params]:
      agent._rng, rng_input = jax.random.split(agent._rng)
      output = agent.network_def.apply(params,
                                       self.ones_state,
                                       num_quantiles=agent.num_quantile_samples,
                                       rng=rng_input)
      self.assertEqual(output.quantile_values.shape[0],
                       agent.num_quantile_samples)
      self.assertEqual(output.quantiles.shape[0],
                       agent.num_quantile_samples)
      self.assertEqual(output.quantiles.shape[1], 1)
    # Check the setting of num_actions.
    self.assertEqual(self._num_actions, agent.num_actions)

  def testQValueComputation(self):
    agent = self._create_test_agent()
    quantiles = onp.ones(agent.num_quantile_samples)
    q_value = onp.sum(quantiles)
    quantiles = quantiles.reshape([agent.num_quantile_samples, 1])
    expected_q_values = onp.ones([agent.num_actions]) * q_value
    for params in [agent.online_params, agent.target_network_params]:
      agent._rng, rng_input = jax.random.split(agent._rng)
      q_values = jnp.mean(
          agent.network_def.apply(
              params, self.ones_state, num_quantiles=agent.num_quantile_samples,
              rng=rng_input).quantile_values, axis=0)
      onp.array_equal(q_values, expected_q_values)
      self.assertEqual(jnp.argmax(q_values, axis=0), 0)

  def testReplayQuantileValueShape(self):
    agent = self._create_test_agent()
    batch_size = 32
    batch_states = onp.ones(
        (batch_size,) + self.observation_shape + (self.stack_size,))
    for params in [agent.online_params, agent.target_network_params]:
      agent._rng, rng_input = jax.random.split(agent._rng)
      model_output = jax.vmap(
          lambda n, p, x, y, z: n.apply(p, x=x, num_quantiles=y, rng=z),
          in_axes=(None, None, 0, None, None))(
              agent.network_def, params, batch_states, agent.num_tau_samples,
              rng_input)
      quantile_values = model_output.quantile_values
      quantile_values = jnp.squeeze(quantile_values)
      self.assertEqual(quantile_values.shape[0], batch_size)
      self.assertEqual(quantile_values.shape[1], agent.num_tau_samples)
      self.assertEqual(quantile_values.shape[2], agent.num_actions)

  def testBeginEpisode(self):
    """Test the functionality of agent.begin_episode.

    Specifically, the action returned and its effect on state.
    """
    agent = self._create_test_agent()
    # We fill up the state with 9s. On calling agent.begin_episode the state
    # should be reset to all 0s.
    agent.state.fill(9)
    first_observation = onp.ones(self.observation_shape + (1,))
    self.assertEqual(agent.begin_episode(first_observation), 0)
    # When the all-1s observation is received, it will be placed at the end of
    # the state.
    expected_state = onp.zeros(self.observation_shape + (self.stack_size,))
    expected_state[:, :, -1] = onp.ones(self.observation_shape)
    self.assertTrue(onp.array_equal(agent.state, expected_state))
    self.assertTrue(onp.array_equal(agent._observation,
                                    first_observation[:, :, 0]))
    # No training happens in eval mode.
    self.assertEqual(agent.training_steps, 0)
    # This will now cause training to happen.
    agent.eval_mode = False
    # Having a low replay memory add_count will prevent any of the
    # train/prefetch/sync ops from being called.
    agent._replay.add_count = 0
    second_observation = onp.ones(self.observation_shape + (1,)) * 2
    agent.begin_episode(second_observation)
    # The agent's state will be reset, so we will only be left with the all-2s
    # observation.
    expected_state[:, :, -1] = onp.full(self.observation_shape, 2)
    self.assertTrue(onp.array_equal(agent.state, expected_state))
    self.assertTrue(onp.array_equal(agent._observation,
                                    second_observation[:, :, 0]))
    # training_steps is incremented since we set eval_mode to False.
    self.assertEqual(agent.training_steps, 1)


if __name__ == '__main__':
  absltest.main()

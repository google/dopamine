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
"""Tests for dopamine.jax.agents.quantile_agent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from absl.testing import absltest
from dopamine.discrete_domains import atari_lib
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.agents.quantile import quantile_agent
from dopamine.utils import test_utils
from flax import linen
import gin
import jax.numpy as jnp
import numpy as onp


class JaxQuantileAgentTest(absltest.TestCase):

  def setUp(self):
    super(JaxQuantileAgentTest, self).setUp()
    self.num_actions = 4
    self._num_atoms = 5
    self._min_replay_history = 32
    self._epsilon_decay_period = 90
    self.observation_shape = dqn_agent.NATURE_DQN_OBSERVATION_SHAPE
    self.observation_dtype = dqn_agent.NATURE_DQN_DTYPE
    self.stack_size = dqn_agent.NATURE_DQN_STACK_SIZE
    self.zero_state = onp.zeros(
        (1,) + self.observation_shape + (self.stack_size,))
    gin.bind_parameter('OutOfGraphPrioritizedReplayBuffer.replay_capacity', 100)
    gin.bind_parameter('OutOfGraphPrioritizedReplayBuffer.batch_size', 2)

  def _create_test_agent(self):
    """Keras network for tests."""
    # This dummy network allows us to deterministically anticipate that
    # action 0 will be selected by an argmax.

    # In Quantile we are dealing with a distribution over Q-values,
    # which are represented as num_atoms quantiles.
    # The output layer will have num_actions * num_atoms elements,
    # so each group of num_atoms weights represent the value quantiles for
    # a particular action. By setting 1s everywhere, except for the first
    # num_atoms (representing the quantiles for the first action), which
    # are set to onp.arange(num_atoms), we are ensuring that the first action
    # has a higher expected Q-value; this results in the first
    # action being chosen.
    class MockQuantileNetwork(linen.Module):
      """Custom Jax network used in tests."""
      num_actions: int
      num_atoms: int
      inputs_preprocessed: bool = False

      @linen.compact
      def __call__(self, x):
        def custom_init(key, shape, dtype=jnp.float32):
          del key
          to_pick_first_action = onp.ones(shape, dtype)
          to_pick_first_action[:, :self.num_atoms] = onp.arange(
              1, self.num_atoms + 1)
          return to_pick_first_action

        x = x.astype(jnp.float32)
        x = x.reshape((-1))  # flatten
        x = linen.Dense(features=self.num_actions * self.num_atoms,
                        kernel_init=custom_init,
                        bias_init=linen.initializers.ones)(x)
        logits = x.reshape((self.num_actions, self.num_atoms))
        probabilities = linen.softmax(logits)
        qs = jnp.mean(logits, axis=1)
        return atari_lib.RainbowNetworkType(qs, logits, probabilities)

    agent = quantile_agent.JaxQuantileAgent(
        network=MockQuantileNetwork,
        num_actions=self.num_actions,
        num_atoms=self._num_atoms,
        min_replay_history=self._min_replay_history,
        epsilon_fn=lambda w, x, y, z: 0.0,  # No exploration.
        epsilon_eval=0.0,
        epsilon_decay_period=self._epsilon_decay_period)
    # This ensures non-random action choices (since epsilon_eval = 0.0) and
    # skips the train_step.
    agent.eval_mode = True
    return agent

  def testCreateAgentWithDefaults(self):
    # Verifies that we can create and train an agent with the default values.
    agent = quantile_agent.JaxQuantileAgent(num_actions=4)
    observation = onp.ones([84, 84, 1])
    agent.begin_episode(observation)
    agent.step(reward=1, observation=observation)
    agent.end_episode(reward=1)

  def testShapesAndValues(self):
    agent = self._create_test_agent()
    state = onp.ones((1, 28224))
    net_outputs = agent.network_def.apply(agent.online_params, state)
    self.assertEqual(net_outputs.logits.shape,
                     (self.num_actions, self._num_atoms))
    self.assertEqual(net_outputs.probabilities.shape,
                     net_outputs.logits.shape)
    self.assertEqual(net_outputs.q_values.shape,
                     (self.num_actions,))
    # Check probabilities are uniform for all except the first action.
    expected_probabilities = (
        onp.ones_like(net_outputs.probabilities) * 1.0 / self._num_atoms)
    # The first action will have probability 1.0 at the highest quantile (it is
    # set this way by design in the mock network to guarantee that action 1 is
    # always selected).
    expected_probabilities[0] = onp.zeros(self._num_atoms)
    expected_probabilities[0, -1] = 1.0
    onp.testing.assert_allclose(net_outputs.probabilities,
                                expected_probabilities)

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
    expected_state = self.zero_state
    expected_state[:, :, :, -1] = onp.ones((1,) + self.observation_shape)
    onp.array_equal(agent.state, expected_state)
    onp.array_equal(agent._observation, first_observation[:, :, 0])
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
    expected_state[:, :, :, -1] = onp.full((1,) + self.observation_shape, 2)
    onp.array_equal(agent.state, expected_state)
    onp.array_equal(agent._observation, second_observation[:, :, 0])
    # training_steps is incremented since we set eval_mode to False.
    self.assertEqual(agent.training_steps, 1)

  def testStepEval(self):
    """Test the functionality of agent.step() in eval mode.

    Specifically, the action returned, and confirm no training is happening.
    """
    agent = self._create_test_agent()
    base_observation = onp.ones(self.observation_shape + (1,))
    # This will reset state and choose a first action.
    agent.begin_episode(base_observation)
    # We mock the replay buffer to verify how the agent interacts with it.
    agent._replay = test_utils.MockReplayBuffer()

    expected_state = self.zero_state
    num_steps = 10
    for step in range(1, num_steps + 1):
      # We make observation a multiple of step for testing purposes (to
      # uniquely identify each observation).
      observation = base_observation * step
      self.assertEqual(agent.step(reward=1, observation=observation), 0)
      stack_pos = step - num_steps - 1
      if stack_pos >= -self.stack_size:
        expected_state[:, :, :, stack_pos] = onp.full(
            (1,) + self.observation_shape, step)
    onp.array_equal(agent.state, expected_state)
    onp.array_equal(
        agent._last_observation,
        onp.ones(self.observation_shape) * (num_steps - 1))
    onp.array_equal(agent._observation, observation[:, :, 0])
    # No training happens in eval mode.
    self.assertEqual(agent.training_steps, 0)
    # No transitions are added in eval mode.
    self.assertEqual(agent._replay.add.call_count, 0)

  def testStepTrain(self):
    """Test the functionality of agent.step() in train mode.

    Specifically, the action returned, and confirm training is happening.
    """
    agent = self._create_test_agent()
    agent.eval_mode = False
    base_observation = onp.ones(self.observation_shape + (1,))
    # We mock the replay buffer to verify how the agents interacts with it.
    agent._replay = test_utils.MockReplayBuffer(is_jax=True)
    # This will reset state and choose a first action.
    agent.begin_episode(base_observation)
    expected_state = self.zero_state
    num_steps = 10
    for step in range(1, num_steps + 1):
      # We make observation a multiple of step for testing purposes (to
      # uniquely identify each observation).
      observation = base_observation * step
      self.assertEqual(agent.step(reward=1, observation=observation), 0)
      stack_pos = step - num_steps - 1
      if stack_pos >= -self.stack_size:
        expected_state[:, :, :, stack_pos] = onp.full(
            (1,) + self.observation_shape, step)
    onp.array_equal(agent.state, expected_state)
    onp.array_equal(
        agent._last_observation,
        onp.full(self.observation_shape, num_steps - 1))
    onp.array_equal(agent._observation, observation[:, :, 0])
    # We expect one more than num_steps because of the call to begin_episode.
    onp.array_equal(agent.training_steps, num_steps + 1)
    onp.array_equal(agent._replay.add.call_count, num_steps)
    agent.end_episode(reward=1)
    self.assertEqual(agent._replay.add.call_count, num_steps + 1)


if __name__ == '__main__':
  absltest.main()

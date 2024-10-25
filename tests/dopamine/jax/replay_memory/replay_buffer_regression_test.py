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
"""Regression tests for replay_buffer.py.

These tests are based on the tests for the original circular replay buffer, and
are meant to ensure the new buffer behaves as expected.
"""

from absl.testing import absltest
from absl.testing import parameterized
from dopamine.jax.replay_memory import accumulator
from dopamine.jax.replay_memory import elements
from dopamine.jax.replay_memory import replay_buffer
from dopamine.jax.replay_memory import samplers
import numpy as np


# Default parameters used when creating the replay memory.
OBSERVATION_SHAPE = (84, 84)
OBS_DTYPE = np.uint8
STACK_SIZE = 4
BATCH_SIZE = 32


class ReplayBufferRegressionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    num_dims = 10
    self._test_observation = np.ones(num_dims) * 1
    self._test_action = np.ones(num_dims) * 2
    self._test_reward = np.ones(num_dims) * 3
    self._test_terminal = np.ones(num_dims) * 4
    self._test_add_count = np.array(7)
    self._transition_accumulator = accumulator.TransitionAccumulator(
        stack_size=STACK_SIZE,
        update_horizon=5,
        gamma=1.0,
    )
    self._sampling_distribution = samplers.UniformSamplingDistribution(seed=0)

  def testNSteprewards(self):
    memory = replay_buffer.ReplayBuffer(
        transition_accumulator=self._transition_accumulator,
        sampling_distribution=self._sampling_distribution,
        batch_size=BATCH_SIZE,
        max_capacity=10,
        compress=False,
    )

    for i in range(50):
      transition = elements.TransitionElement(
          np.full(OBSERVATION_SHAPE, i, dtype=OBS_DTYPE), 0, 2.0, False
      )
      memory.add(transition)

    for _ in range(100):
      batch = memory.sample()
      # Make sure the total reward is reward per step x update_horizon.
      np.testing.assert_array_equal(batch.reward, np.ones(BATCH_SIZE) * 10.0)

  def testGetStack(self):
    zero_state = np.zeros(OBSERVATION_SHAPE + (3,), dtype=OBS_DTYPE)

    memory = replay_buffer.ReplayBuffer(
        transition_accumulator=self._transition_accumulator,
        sampling_distribution=self._sampling_distribution,
        batch_size=BATCH_SIZE,
        max_capacity=50,
        compress=False,
    )
    for i in range(11):
      transition = elements.TransitionElement(
          np.full(OBSERVATION_SHAPE, i, dtype=OBS_DTYPE), 0, 0, False
      )
      memory.add(transition)

    # ensure that the returned shapes are always correct
    for i in memory._memory:
      np.testing.assert_array_equal(
          memory._memory[i].state.shape, OBSERVATION_SHAPE + (4,)
      )

    # ensure that there is the necessary 0 padding
    state = memory._memory[0].state
    np.testing.assert_array_equal(zero_state, state[:, :, :3])

    # ensure that after the padding the contents are properly stored
    state = memory._memory[3].state
    for i in range(4):
      np.testing.assert_array_equal(
          np.full(OBSERVATION_SHAPE, i), state[:, :, i]
      )

  def testSampleTransitionBatch(self):
    replay_capacity = 10
    transition_accumulator = accumulator.TransitionAccumulator(
        stack_size=1,
        update_horizon=1,
        gamma=0.99,
    )
    memory = replay_buffer.ReplayBuffer(
        transition_accumulator=transition_accumulator,
        sampling_distribution=self._sampling_distribution,
        batch_size=2,
        max_capacity=replay_capacity,
    )
    num_adds = 50  # The number of transitions to add to the memory.
    # Since terminal transitions are not valid trajectories (since there is no
    # next state), they will not be sent back to the replay buffer memory.
    # Thus, we do not have a one-to-one correspondence between indices and the
    # value contained in the respective state. We use this map to keep track of
    # the mapping from indices to sample and state value.
    index_to_id = []
    for i in range(num_adds):
      terminal = i % 4 == 0  # Every 4 transitions is terminal.
      transition = elements.TransitionElement(
          np.full(OBSERVATION_SHAPE, i, OBS_DTYPE), 0, 0, terminal, False
      )
      memory.add(transition)
      if not terminal:
        index_to_id.append(i)
    # Test sampling with default batch size.
    for _ in range(1000):
      batch = memory.sample()
      self.assertEqual(batch.state.shape[0], 2)
    # Test changing batch sizes.
    for _ in range(1000):
      batch = memory.sample(BATCH_SIZE)
      self.assertEqual(batch.state.shape[0], BATCH_SIZE)
    # Verify we revert to default batch size.
    for _ in range(1000):
      batch = memory.sample()
      self.assertEqual(batch.state.shape[0], 2)

    sampler = memory._sampling_distribution
    self.assertIsInstance(sampler, samplers.UniformSamplingDistribution)

    # Verify we sample the expected indices.
    # Use the same rng state for reprodcibility.
    rng = np.random.default_rng(0)
    rng.bit_generator.state = sampler._rng.bit_generator.state
    indices = rng.integers(
        len(sampler._key_by_index), size=len(sampler._key_by_index)
    )

    def make_state(key: int) -> np.ndarray:
      return np.full(OBSERVATION_SHAPE + (1,), key, dtype=OBS_DTYPE)

    expected_states = np.array(
        [make_state(index_to_id[sampler._key_by_index[i]]) for i in indices]
    )
    expected_next_states = np.array(
        [make_state(index_to_id[sampler._key_by_index[i]] + 1) for i in indices]
    )
    # This is replicating the formula that was used above to determine what
    # transitions are terminal when adding observation (i % 4). However, we add
    # 1 to the index, as it is the _next_ state that determines whether this was
    # a terminal transition or not.
    expected_terminal = np.array([
        int(((index_to_id[sampler._key_by_index[i]] + 1) % 4) == 0)
        for i in indices
    ])
    batch = memory.sample(size=len(indices))
    np.testing.assert_array_equal(batch.state, expected_states)
    np.testing.assert_array_equal(batch.action, np.zeros(len(indices)))
    np.testing.assert_array_equal(batch.reward, np.zeros(len(indices)))
    np.testing.assert_array_equal(batch.next_state, expected_next_states)
    np.testing.assert_array_equal(batch.is_terminal, expected_terminal)

  def testSamplingWithTerminalInTrajectory(self):
    replay_capacity = 10
    update_horizon = 3
    transition_accumulator = accumulator.TransitionAccumulator(
        stack_size=1,
        update_horizon=update_horizon,
        gamma=1.0,
    )
    memory = replay_buffer.ReplayBuffer(
        transition_accumulator=transition_accumulator,
        sampling_distribution=self._sampling_distribution,
        batch_size=2,
        max_capacity=replay_capacity,
    )
    for i in range(replay_capacity):
      transition = elements.TransitionElement(
          np.full(OBSERVATION_SHAPE, i, OBS_DTYPE),
          i * 2,  # action
          i,  # reward
          i == 3,  # terminal
          False,  # episode_end
      )
      memory.add(transition)
    # Verify we sample the expected indices, using the same rng.
    rng = np.random.default_rng(0)
    indices = rng.integers(memory.add_count, size=(5,))
    batch = memory.sample(size=5)
    # Since index 3 is terminal, it will not be a valid transition so we need to
    # "renumber" accordingly.
    expected_states = np.array([
        np.full(OBSERVATION_SHAPE + (1,), i, dtype=OBS_DTYPE)
        if i < 3
        else np.full(OBSERVATION_SHAPE + (1,), i + 1, dtype=OBS_DTYPE)
        for i in indices
    ])
    expected_actions = np.array(
        [i * 2 if i < 3 else (i + 1) * 2 for i in indices]
    )
    # The reward in the replay buffer will be (an asterisk marks the terminal
    # state):
    #   [0 1 2 3* 4 5 6 7 8 9]
    # Since we're setting the update_horizon to 3, the accumulated trajectory
    # reward starting at each of the replay buffer positions will be (a '_'
    # marks an invalid transition to sample):
    #   [3 6 5 _ 15 18 21 24]
    expected_rewards = np.array([3, 6, 5, 15, 18, 21, 24])
    # Because update_horizon = 3, indices 0, 1 and 2 include terminal.
    expected_terminals = np.array([1, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(batch.state, expected_states)
    np.testing.assert_array_equal(batch.action, expected_actions)
    np.testing.assert_array_equal(batch.reward, expected_rewards[indices])
    np.testing.assert_array_equal(
        batch.is_terminal, expected_terminals[indices]
    )


if __name__ == '__main__':
  absltest.main()

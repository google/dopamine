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
"""Tests for replay_buffer.py.

The actual replay buffer doesn't have much surface area as it depends on
a sampler and accumulator and these are both tested independently.
We only test the legacy replay buffer here which validates the integration
of the sampler and accumulator.
"""

from absl.testing import absltest
from absl.testing import parameterized
from dopamine.jax.replay_memory import accumulator
from dopamine.jax.replay_memory import elements
from dopamine.jax.replay_memory import replay_buffer
from dopamine.jax.replay_memory import samplers
from etils import epath
import msgpack
import numpy as np

mock = absltest.mock


class ReplayBufferTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._tmpdir = epath.Path(self.create_tempdir('checkpoint').full_path)
    self._obs = np.ones((4, 3))
    self._transition_accumulator = accumulator.TransitionAccumulator(
        stack_size=2,
        update_horizon=10,
        gamma=0.9,
    )
    self._sampling_distribution = samplers.UniformSamplingDistribution(seed=0)
    self._sample_transition = elements.TransitionElement(
        self._obs, 1, 3.14, False, False
    )

  @parameterized.parameters((-1,), (0,))
  def testWithInvalidCheckpointDuration(self, cd):
    with self.assertRaises(ValueError):
      _ = replay_buffer.ReplayBuffer(
          transition_accumulator=self._transition_accumulator,
          sampling_distribution=self._sampling_distribution,
          batch_size=32,
          max_capacity=100,
          checkpoint_duration=cd,
      )

  def testCreateReplayBuffer(self):
    _ = replay_buffer.ReplayBuffer(
        transition_accumulator=self._transition_accumulator,
        sampling_distribution=self._sampling_distribution,
        batch_size=32,
        max_capacity=100,
        compress=True,
    )

  @mock.patch.object(accumulator, 'TransitionAccumulator', autospec=True)
  def testAddWithEmptyAccumulatorReturn(self, mock_accumulator):
    instance = mock_accumulator.return_value
    instance.accumulate.return_value = []
    replay = replay_buffer.ReplayBuffer(
        transition_accumulator=instance,
        sampling_distribution=self._sampling_distribution,
        batch_size=32,
        max_capacity=100,
    )
    replay.add(self._sample_transition)
    instance.accumulate.assert_called_with(self._sample_transition)
    self.assertEmpty(replay._memory)

  @mock.patch.object(accumulator, 'TransitionAccumulator', autospec=True)
  def testAddWithoutCompress(self, mock_accumulator):
    instance = mock_accumulator.return_value
    instance.accumulate.return_value = []
    replay = replay_buffer.ReplayBuffer(
        transition_accumulator=instance,
        sampling_distribution=self._sampling_distribution,
        batch_size=32,
        max_capacity=100,
        compress=False,
    )
    replay.add(self._sample_transition)
    instance.accumulate.assert_called_with(self._sample_transition)
    self.assertEmpty(replay._memory)

  @mock.patch.object(accumulator, 'TransitionAccumulator', autospec=True)
  def testAddWithValidAccumulatorReturn(self, mock_accumulator):
    instance = mock_accumulator.return_value
    instance.accumulate.return_value = [self._sample_transition]
    replay = replay_buffer.ReplayBuffer(
        transition_accumulator=instance,
        sampling_distribution=self._sampling_distribution,
        batch_size=32,
        max_capacity=100,
        compress=False,
    )
    replay.add(self._sample_transition)
    self.assertLen(replay._memory, 1)
    self.assertEqual(list(replay._memory.keys()), [0])
    np.testing.assert_array_equal(
        replay._memory[0].observation, self._sample_transition.observation
    )
    self.assertEqual(replay._memory[0].action, self._sample_transition.action)
    self.assertEqual(replay._memory[0].reward, self._sample_transition.reward)
    self.assertEqual(
        replay._memory[0].is_terminal, int(self._sample_transition.is_terminal)
    )
    self.assertEqual(
        replay._memory[0].episode_end, int(self._sample_transition.episode_end)
    )

  @mock.patch.object(accumulator, 'TransitionAccumulator', autospec=True)
  def testAddUpToCapacity(self, mock_accumulator):
    instance = mock_accumulator.return_value
    capacity = 10
    replay = replay_buffer.ReplayBuffer(
        transition_accumulator=instance,
        sampling_distribution=self._sampling_distribution,
        batch_size=32,
        max_capacity=capacity,
        compress=False,
    )
    transitions = []
    for i in range(15):
      transitions.append(
          elements.TransitionElement(self._obs * i, i, i, False, False)
      )
      instance.accumulate.return_value = [transitions[-1]]
      replay.add(transitions[-1])
    # Since we created the ReplayBuffer with a capacity of 10, it should have
    # gotten rid of the first 5 elements added.
    self.assertLen(replay._memory, capacity)
    expected_keys = list(range(5, 5 + capacity))
    self.assertEqual(list(replay._memory.keys()), expected_keys)
    for i in expected_keys:
      np.testing.assert_array_equal(
          replay._memory[i].observation, transitions[i].observation
      )
      self.assertEqual(replay._memory[i].action, transitions[i].action)
      self.assertEqual(replay._memory[i].reward, transitions[i].reward)
      self.assertEqual(
          replay._memory[i].is_terminal, int(transitions[i].is_terminal)
      )
      self.assertEqual(
          replay._memory[i].episode_end, int(transitions[i].episode_end)
      )

  def testSampleWithNoElements(self):
    replay = replay_buffer.ReplayBuffer(
        transition_accumulator=self._transition_accumulator,
        sampling_distribution=self._sampling_distribution,
        batch_size=32,
        max_capacity=100,
    )
    with self.assertRaises(ValueError):
      replay.sample()
    with self.assertRaises(ValueError):
      replay.sample(512)

  @parameterized.named_parameters(
      dict(
          testcase_name='DefaultBSNoCompress',
          use_default_bs=True,
          compress=False,
      ),
      dict(
          testcase_name='NoDefaultBSNoCompress',
          use_default_bs=False,
          compress=False,
      ),
      dict(
          testcase_name='DefaultBSCompress', use_default_bs=True, compress=True
      ),
      dict(
          testcase_name='NoDefaultBSCompress',
          use_default_bs=False,
          compress=True,
      ),
  )
  @mock.patch.object(accumulator, 'TransitionAccumulator', autospec=True)
  def testSample(self, mock_accumulator, use_default_bs, compress):
    instance = mock_accumulator.return_value
    seed = 0
    capacity = 100
    default_batch_size = 32
    replay = replay_buffer.ReplayBuffer(
        transition_accumulator=instance,
        sampling_distribution=self._sampling_distribution,
        batch_size=default_batch_size,
        max_capacity=capacity,
        compress=compress,
    )
    transitions = []
    for i in range(100):
      transitions.append(
          elements.TransitionElement(self._obs * i, i, i, False, False)
      )
      state = transitions[-1].observation
      next_state = self._obs * (i + 1)
      instance.accumulate.return_value = [
          elements.ReplayElement(
              state=state,
              next_state=next_state,
              action=i,  # action
              reward=i,  # reward
              is_terminal=int(transitions[-1].is_terminal),
              episode_end=int(transitions[-1].episode_end),
          )
      ]
      replay.add(transitions[-1])
    expected_keys = list(range(100))
    self.assertEqual(list(replay._memory.keys()), expected_keys)
    batch_size = default_batch_size if use_default_bs else 16
    rng = np.random.default_rng(seed)  # Use the same numpy RNG.
    if use_default_bs:
      batches = replay.sample()
    else:
      batches = replay.sample(size=batch_size)
    self.assertEqual(batches.state.shape[0], batch_size)
    self.assertEqual(batches.next_state.shape[0], batch_size)
    self.assertEqual(batches.action.shape[0], batch_size)
    self.assertEqual(batches.reward.shape[0], batch_size)
    self.assertLen(batches.is_terminal, batch_size)
    self.assertLen(batches.episode_end, batch_size)

    # Reset numpy random state.
    indices = rng.integers(100, size=(batch_size,))
    for i, idx in enumerate(indices):
      np.testing.assert_array_equal(
          batches.state[i], transitions[idx].observation
      )
      np.testing.assert_array_equal(
          batches.next_state[i], np.ones_like(self._obs) * (idx + 1)
      )
      self.assertEqual(batches.action[i], idx)
      self.assertEqual(batches.reward[i], idx)
      self.assertEqual(batches.is_terminal[i], 0)
      self.assertEqual(batches.episode_end[i], 0)

  @parameterized.parameters((True,), (False,))
  def testSave(self, compress):
    stack_size = 4
    replay_capacity = 50
    batch_size = 32
    update_horizon = 3
    gamma = 0.9
    checkpoint_duration = 7
    transition_accumulator = accumulator.TransitionAccumulator(
        stack_size=stack_size,
        update_horizon=update_horizon,
        gamma=gamma,
    )
    replay = replay_buffer.ReplayBuffer(
        transition_accumulator=transition_accumulator,
        sampling_distribution=self._sampling_distribution,
        batch_size=batch_size,
        max_capacity=replay_capacity,
        checkpoint_duration=checkpoint_duration,
        compress=compress,
    )
    # Store a few transitions in memory. Since update_horizon is 3, only
    # num_adds - 3 elements will actually be in the replay buffer memory.
    transitions = []
    num_adds = 15
    for i in range(num_adds):
      transitions.append(
          elements.TransitionElement(self._obs * i, i, i, False, False)
      )
      replay.add(transitions[-1])

    replay.save(self._tmpdir, 1)
    path = self._tmpdir / '1' / 'replay' / 'checkpoint.msgpack'
    self.assertTrue(path.exists())
    replay_pack = msgpack.unpackb(
        path.read_bytes(),
        raw=False,
        strict_map_key=False,
    )
    self.assertEqual(num_adds - update_horizon, replay_pack['add_count'])

  @parameterized.parameters((1,), (3,), (5,))
  def testGarbageCollection(self, cd):
    replay = replay_buffer.ReplayBuffer(
        transition_accumulator=self._transition_accumulator,
        sampling_distribution=self._sampling_distribution,
        batch_size=8,
        max_capacity=10,
        checkpoint_duration=cd,
    )
    num_adds = 20
    # Store transitions and call save each time.
    for i in range(num_adds):
      replay.add(elements.TransitionElement(self._obs * i, i, i, False, False))
      replay.save(self._tmpdir, i)
    for i in range(num_adds):
      path = self._tmpdir / f'{i}' / 'replay'
      if i < num_adds - cd:
        self.assertFalse(path.exists())
      else:
        self.assertTrue(path.exists())

  def testLoad(self):
    update_horizon = 3
    transition_accumulator = accumulator.TransitionAccumulator(
        stack_size=4,
        update_horizon=update_horizon,
        gamma=0.99,
    )
    replay = replay_buffer.ReplayBuffer(
        transition_accumulator=transition_accumulator,
        sampling_distribution=self._sampling_distribution,
        batch_size=8,
        max_capacity=10,
        checkpoint_duration=10,
        compress=False,
    )
    num_adds = 20
    # Store transitions and call save each time.
    for i in range(num_adds):
      replay.add(elements.TransitionElement(self._obs * i, i, i, False, False))
      replay.save(self._tmpdir, i)
    # add_count should be equal to num_adds - update_horizon now.
    self.assertEqual(num_adds - update_horizon, replay.add_count)
    # Load a checkpoint from 5 iterations ago (it is zero-indexed so we
    # substract 6).
    replay.load(self._tmpdir, num_adds - 6)
    # add_count should now be equal to num_adds - update_horizon  - 5.
    self.assertEqual(num_adds - update_horizon - 5, replay.add_count)

  def testSaveAndLoad(self):
    # This is similar to testSample, but saves and loads after every addition.
    # The resulting behaviour should be identical.
    capacity = 100
    batch_size = 32
    transition_accumulator = accumulator.TransitionAccumulator(
        stack_size=2,
        update_horizon=1,
        gamma=0.99,
    )
    replay = replay_buffer.ReplayBuffer(
        transition_accumulator=transition_accumulator,
        sampling_distribution=self._sampling_distribution,
        batch_size=batch_size,
        max_capacity=capacity,
    )
    transitions = []
    for i in range(100):
      transitions.append(
          elements.TransitionElement(self._obs * i, i, i, False, False)
      )
      replay.add(transitions[-1])
      # The first call to add() will not actually produce anything in the
      # memory, as there are not yet any valid transitions.
      if i > 0:
        _ = replay.sample()
      replay.save(self._tmpdir, i)
      # We reinitialize the memory to ensure it's loading properly.
      replay = replay_buffer.ReplayBuffer(
          transition_accumulator=transition_accumulator,
          sampling_distribution=self._sampling_distribution,
          batch_size=batch_size,
          max_capacity=capacity,
      )
      replay.load(self._tmpdir, i)
    expected_keys = list(range(99))  # The last transition is invalid.
    self.assertEqual(list(replay._memory.keys()), expected_keys)
    # Save the rng state before sampling for reprodcibility.
    rng_state = replay._sampling_distribution._rng.bit_generator.state
    batches = replay.sample()
    self.assertEqual(batches.state.shape[0], batch_size)
    self.assertEqual(batches.next_state.shape[0], batch_size)
    self.assertEqual(batches.action.shape[0], batch_size)
    self.assertEqual(batches.reward.shape[0], batch_size)
    self.assertLen(batches.is_terminal, batch_size)
    self.assertLen(batches.episode_end, batch_size)

    rng = np.random.default_rng(0)
    # Set the state to that of the replay before sampling.
    rng.bit_generator.state = rng_state
    indices = rng.choice(99, size=(batch_size,))
    for i, idx in enumerate(indices):
      # We need to stack the observations for proper comparison.
      if idx == 0:
        stacked_state = [np.zeros_like(self._obs), transitions[idx].observation]
      else:
        stacked_state = [
            transitions[idx - 1].observation,
            transitions[idx].observation,
        ]
      stacked_next_state = [
          transitions[idx].observation,
          transitions[idx + 1].observation,
      ]
      stacked_state = np.moveaxis(stacked_state, 0, -1)
      stacked_next_state = np.moveaxis(stacked_next_state, 0, -1)
      np.testing.assert_array_equal(batches.state[i], stacked_state)
      np.testing.assert_array_equal(batches.next_state[i], stacked_next_state)
      self.assertEqual(batches.action[i], idx)
      self.assertEqual(batches.reward[i], idx)
      self.assertEqual(batches.is_terminal[i], 0)
      self.assertEqual(batches.episode_end[i], 0)

  def testKeyMappingsForSampling(self):
    capacity = 10
    batch_size = 32
    transition_accumulator = accumulator.TransitionAccumulator(
        stack_size=1,
        update_horizon=1,
        gamma=0.99,
    )
    replay = replay_buffer.ReplayBuffer(
        transition_accumulator=transition_accumulator,
        sampling_distribution=self._sampling_distribution,
        batch_size=batch_size,
        max_capacity=capacity,
    )
    sampler = replay._sampling_distribution

    for i in range(capacity + 1):
      replay.add(elements.TransitionElement(self._obs * i, i, i, False, False))

    # While we haven't overwritten any elements we should have
    # global indices as being equivalent to local indices
    for i in range(capacity):
      self.assertIn(i, sampler._index_by_key)
      index = sampler._index_by_key[i]
      self.assertEqual(i, index)
      self.assertEqual(i, sampler._key_by_index[index])

    # The next key to be inserted will be `capacity` as when we add
    # `capacity + 1` the accumulator will insert: (capacity, capacity + 1)
    next_key = capacity
    replay.add(
        elements.TransitionElement(
            self._obs * (next_key + 1),
            next_key + 1,
            next_key + 1,
            False,
            False,
        )
    )
    # We should have deleted the earliest index
    self.assertNotIn(0, sampler._index_by_key)
    # The local index corresponding to the previous key should have been swapped
    self.assertNotEqual(sampler._key_by_index[0], 0)
    # We should have inserted the new key into key -> index
    self.assertIn(next_key, sampler._index_by_key)
    # index -> key should be consistent
    self.assertEqual(
        next_key, sampler._key_by_index[sampler._index_by_key[next_key]]
    )

    # Sampling
    rng = np.random.default_rng(0)
    # Set the state to that of the replay before sampling.
    rng.bit_generator.state = sampler._rng.bit_generator.state
    # Sample local indices
    indices = rng.integers(len(sampler._key_by_index), size=(batch_size,))
    # Convert local indices to global keys
    keys = (sampler._key_by_index[index] for index in indices)

    # Fetch actual samples from the replay buffer so we can compare
    # the global indices
    samples = replay.sample()

    # Each index in our samples should have observations that are equal to
    # their global key, we can check this:
    for i, key in enumerate(keys):
      np.testing.assert_array_equal(
          samples.state[i, ...],
          self._obs[..., None] * key,
      )
      np.testing.assert_array_equal(
          samples.next_state[i, ...],
          self._obs[..., None] * (key + 1),
      )
      self.assertEqual(samples.action[i], key)
      self.assertEqual(samples.reward[i], key)
      self.assertEqual(samples.is_terminal[i], 0)
      self.assertEqual(samples.episode_end[i], 0)

  def testClearBuffer(self):
    capacity = 10
    batch_size = 32
    transition_accumulator = accumulator.TransitionAccumulator(
        stack_size=1,
        update_horizon=1,
        gamma=0.99,
    )
    replay = replay_buffer.ReplayBuffer(
        transition_accumulator=transition_accumulator,
        sampling_distribution=self._sampling_distribution,
        batch_size=batch_size,
        max_capacity=capacity,
    )
    for i in range(capacity):
      replay.add(elements.TransitionElement(self._obs * i, i, i, False, False))
    self.assertNotEmpty(replay._memory)
    self.assertNotEqual(replay.add_count, 0)
    self.assertNotEmpty(replay._transition_accumulator._trajectory)
    self.assertNotEmpty(replay._sampling_distribution._index_by_key)
    self.assertNotEmpty(replay._sampling_distribution._key_by_index)
    replay.clear()
    self.assertEmpty(replay._memory)
    self.assertEqual(replay.add_count, 0)
    self.assertEmpty(replay._transition_accumulator._trajectory)
    self.assertEmpty(replay._sampling_distribution._index_by_key)
    self.assertEmpty(replay._sampling_distribution._key_by_index)


if __name__ == '__main__':
  absltest.main()

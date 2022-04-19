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
"""Tests for circular_replay_buffer.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import shutil



from absl import flags
from dopamine.replay_memory import circular_replay_buffer
import numpy as np
import tensorflow as tf


FLAGS = flags.FLAGS

# Default parameters used when creating the replay memory.
OBSERVATION_SHAPE = (84, 84)
OBS_DTYPE = np.uint8
STACK_SIZE = 4
BATCH_SIZE = 32


class CheckpointableClass(object):

  def __init__(self):
    self.attribute = 0


class OutOfGraphReplayBufferTest(tf.test.TestCase):

  def setUp(self):
    super(OutOfGraphReplayBufferTest, self).setUp()
    self._test_subdir = os.path.join('/tmp/dopamine_tests', 'replay')
    shutil.rmtree(self._test_subdir, ignore_errors=True)
    os.makedirs(self._test_subdir)
    num_dims = 10
    self._test_observation = np.ones(num_dims) * 1
    self._test_action = np.ones(num_dims) * 2
    self._test_reward = np.ones(num_dims) * 3
    self._test_terminal = np.ones(num_dims) * 4
    self._test_add_count = np.array(7)
    self._test_invalid_range = np.ones(num_dims)

  def testWithNontupleObservationShape(self):
    with self.assertRaises(AssertionError):
      _ = circular_replay_buffer.OutOfGraphReplayBuffer(
          observation_shape=84, stack_size=STACK_SIZE, replay_capacity=5,
          batch_size=BATCH_SIZE)

  def testConstructor(self):
    memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=5,
        batch_size=BATCH_SIZE)
    self.assertEqual(memory._observation_shape, OBSERVATION_SHAPE)
    # Test with non square observation shape
    memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=(4, 20),
        stack_size=STACK_SIZE,
        replay_capacity=5,
        batch_size=BATCH_SIZE)
    self.assertEqual(memory._observation_shape, (4, 20))
    self.assertEqual(memory.add_count, 0)
    # Test with terminal datatype of np.int32
    memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        terminal_dtype=np.int32,
        replay_capacity=5,
        batch_size=BATCH_SIZE)
    self.assertEqual(memory._terminal_dtype, np.int32)

  def testAdd(self):
    memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=5,
        batch_size=BATCH_SIZE)
    self.assertEqual(memory.cursor(), 0)
    zeros = np.zeros(OBSERVATION_SHAPE)
    memory.add(zeros, 0, 0, 0)
    # Check if the cursor moved STACK_SIZE -1 padding adds + 1, (the one above).
    self.assertEqual(memory.cursor(), STACK_SIZE)

  def testExtraAdd(self):
    memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=5,
        batch_size=BATCH_SIZE,
        extra_storage_types=[
            circular_replay_buffer.ReplayElement('extra1', [], np.float32),
            circular_replay_buffer.ReplayElement('extra2', [2], np.int8)
        ])
    self.assertEqual(memory.cursor(), 0)
    zeros = np.zeros(OBSERVATION_SHAPE)
    memory.add(zeros, 0, 0, 0, 0, [0, 0])

    with self.assertRaisesRegex(ValueError, 'Add expects'):
      memory.add(zeros, 0, 0, 0)
    # Check if the cursor moved STACK_SIZE -1 zeros adds + 1, (the one above).
    self.assertEqual(memory.cursor(), STACK_SIZE)

  def testCheckAddTypes(self):
    memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=5,
        batch_size=BATCH_SIZE,
        extra_storage_types=[
            circular_replay_buffer.ReplayElement('extra1', [], np.float32),
            circular_replay_buffer.ReplayElement('extra2', [2], np.int8)
        ])
    zeros = np.zeros(OBSERVATION_SHAPE)

    memory._check_add_types(zeros, 0, 0, 0, 0, [0, 0])

    with self.assertRaisesRegex(ValueError, 'Add expects'):
      memory._check_add_types(zeros, 0, 0, 0)

  def testLowCapacity(self):
    with self.assertRaisesRegex(ValueError, 'There is not enough capacity'):
      circular_replay_buffer.OutOfGraphReplayBuffer(
          observation_shape=OBSERVATION_SHAPE,
          stack_size=10,
          replay_capacity=10,
          batch_size=BATCH_SIZE,
          update_horizon=1,
          gamma=1.0)

    with self.assertRaisesRegex(ValueError, 'There is not enough capacity'):
      circular_replay_buffer.OutOfGraphReplayBuffer(
          observation_shape=OBSERVATION_SHAPE,
          stack_size=5,
          replay_capacity=10,
          batch_size=BATCH_SIZE,
          update_horizon=10,
          gamma=1.0)

    # We should be able to create a buffer that contains just enough for a
    # transition.
    circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=5,
        replay_capacity=10,
        batch_size=BATCH_SIZE,
        update_horizon=5,
        gamma=1.0)

  def testGetRangeInvalidIndexOrder(self):
    replay_capacity = 10
    memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=replay_capacity,
        batch_size=BATCH_SIZE,
        update_horizon=5,
        gamma=1.0)
    with self.assertRaisesRegex(AssertionError,
                                'end_index must be larger than start_index'):
      memory.get_range([], 2, 1)
    with self.assertRaises(AssertionError):
      # Negative end_index.
      memory.get_range([], 1, -1)
    with self.assertRaises(AssertionError):
      # Start index beyond replay capacity.
      memory.get_range([], replay_capacity, replay_capacity + 1)
    with self.assertRaisesRegex(AssertionError, 'Index 1 has not been added.'):
      memory.get_range([], 1, 2)

  def testGetRangeNoWraparound(self):
    # Test the get_range function when the indices do not wrap around the
    # circular buffer. In other words, start_index < end_index.
    memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=10,
        batch_size=BATCH_SIZE,
        update_horizon=5,
        gamma=1.0)
    for _ in range(10):
      memory.add(
          np.full(OBSERVATION_SHAPE, 0, dtype=OBS_DTYPE),
          0, 2.0, 0)
    # The constructed `array` will be:
    # array([[ 1.,  1.,  1.,  1.,  1.],
    #        [ 2.,  2.,  2.,  2.,  2.],
    #        [ 3.,  3.,  3.,  3.,  3.],
    #        [ 4.,  4.,  4.,  4.,  4.],
    #        [ 5.,  5.,  5.,  5.,  5.],
    #        [ 6.,  6.,  6.,  6.,  6.],
    #        [ 7.,  7.,  7.,  7.,  7.],
    #        [ 8.,  8.,  8.,  8.,  8.],
    #        [ 9.,  9.,  9.,  9.,  9.],
    #        [10., 10., 10., 10., 10.]])
    array = np.arange(10).reshape(10, 1) + np.ones(5)
    sliced_array = memory.get_range(array, 2, 5)
    self.assertAllEqual(sliced_array, array[2:5])

  def testGetRangeWithWraparound(self):
    # Test the get_range function when the indices wrap around the circular
    # buffer. In other words, start_index > end_index.
    memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=10,
        batch_size=BATCH_SIZE,
        update_horizon=5,
        gamma=1.0)
    for _ in range(10):
      memory.add(
          np.full(OBSERVATION_SHAPE, 0, dtype=OBS_DTYPE),
          0, 2.0, 0)
    # The constructed `array` will be:
    # array([[ 1.,  1.,  1.,  1.,  1.],
    #        [ 2.,  2.,  2.,  2.,  2.],
    #        [ 3.,  3.,  3.,  3.,  3.],
    #        [ 4.,  4.,  4.,  4.,  4.],
    #        [ 5.,  5.,  5.,  5.,  5.],
    #        [ 6.,  6.,  6.,  6.,  6.],
    #        [ 7.,  7.,  7.,  7.,  7.],
    #        [ 8.,  8.,  8.,  8.,  8.],
    #        [ 9.,  9.,  9.,  9.,  9.],
    #        [10., 10., 10., 10., 10.]])
    array = np.arange(10).reshape(10, 1) + np.ones(5)
    sliced_array = memory.get_range(array, 8, 12)
    # We roll by two, since start_index == 8 and replay_capacity == 10, so the
    # resulting indices used will be [8, 9, 0, 1].
    rolled_array = np.roll(array, 2, axis=0)
    self.assertAllEqual(sliced_array, rolled_array[:4])

  def testNSteprewardum(self):
    memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=10,
        batch_size=BATCH_SIZE,
        update_horizon=5,
        gamma=1.0)

    for i in range(50):
      memory.add(
          np.full(OBSERVATION_SHAPE, i, dtype=OBS_DTYPE),
          0, 2.0, 0)

    for i in range(100):
      batch = memory.sample_transition_batch()
      # Make sure the total reward is reward per step x update_horizon.
      self.assertEqual(batch[2][0], 10.0)

  def testGetStack(self):
    zero_stack = np.zeros(OBSERVATION_SHAPE + (4,), dtype=OBS_DTYPE)

    memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=50,
        batch_size=BATCH_SIZE)
    for i in range(11):
      memory.add(
          np.full(OBSERVATION_SHAPE, i, dtype=OBS_DTYPE),
          0, 0, 0)

    # ensure that the returned shapes are always correct
    for i in range(3, memory.cursor()):
      self.assertTrue(
          memory.get_observation_stack(i).shape,
          OBSERVATION_SHAPE + (4,))

    # ensure that there is the necessary 0 padding
    stack = memory.get_observation_stack(3)
    self.assertTrue(np.array_equal(zero_stack, stack))

    # ensure that after the padding the contents are properly stored
    stack = memory.get_observation_stack(6)
    for i in range(4):
      self.assertTrue(
          np.array_equal(np.full(OBSERVATION_SHAPE, i), stack[:, :, i]))

  def testSampleTransitionBatch(self):
    replay_capacity = 10
    memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=1,
        replay_capacity=replay_capacity,
        batch_size=2)
    num_adds = 50  # The number of transitions to add to the memory.
    for i in range(num_adds):
      memory.add(
          np.full(OBSERVATION_SHAPE, i, OBS_DTYPE), 0,
          0, i % 4)  # Every 4 transitions is terminal.
    # Test sampling with default batch size.
    for i in range(1000):
      batch = memory.sample_transition_batch()
      self.assertEqual(batch[0].shape[0], 2)
    # Test changing batch sizes.
    for i in range(1000):
      batch = memory.sample_transition_batch(BATCH_SIZE)
      self.assertEqual(batch[0].shape[0], BATCH_SIZE)
    # Verify we revert to default batch size.
    for i in range(1000):
      batch = memory.sample_transition_batch()
      self.assertEqual(batch[0].shape[0], 2)

    # Verify we can specify what indices to sample.
    indices = [1, 2, 3, 5, 8]
    expected_states = np.array([
        np.full(OBSERVATION_SHAPE + (1,), i, dtype=OBS_DTYPE)
        for i in indices
    ])
    expected_next_states = (expected_states + 1) % replay_capacity
    # Because the replay buffer is circular, we can exactly compute what the
    # states will be at the specified indices by doing a little mod math:
    expected_states += num_adds - replay_capacity
    expected_next_states += num_adds - replay_capacity
    # This is replicating the formula that was used above to determine what
    # transitions are terminal when adding observation (i % 4).
    expected_terminal = np.array(
        [min((x + num_adds - replay_capacity) % 4, 1) for x in indices])
    batch = memory.sample_transition_batch(batch_size=len(indices),
                                           indices=indices)
    (states, action, reward, next_states, next_action, next_reward, terminal,
     indices_batch) = batch
    self.assertAllEqual(states, expected_states)
    self.assertAllEqual(action, np.zeros(len(indices)))
    self.assertAllEqual(reward, np.zeros(len(indices)))
    self.assertAllEqual(next_action, np.zeros(len(indices)))
    self.assertAllEqual(next_reward, np.zeros(len(indices)))
    self.assertAllEqual(next_states, expected_next_states)
    self.assertAllEqual(terminal, expected_terminal)
    self.assertAllEqual(indices_batch, indices)

  def testSampleTransitionBatchExtra(self):
    replay_capacity = 10
    memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=1,
        replay_capacity=replay_capacity,
        batch_size=2,
        extra_storage_types=[
            circular_replay_buffer.ReplayElement('extra1', [], np.float32),
            circular_replay_buffer.ReplayElement('extra2', [2], np.int8)
        ])
    num_adds = 50  # The number of transitions to add to the memory.
    for i in range(num_adds):
      memory.add(
          np.full(OBSERVATION_SHAPE, i, dtype=OBS_DTYPE),
          0, 0, i % 4, 0, [0, 0])  # Every 4 transitions is terminal.
    # Test sampling with default batch size.
    for i in range(1000):
      batch = memory.sample_transition_batch()
      self.assertEqual(batch[0].shape[0], 2)
    # Test changing batch sizes.
    for i in range(1000):
      batch = memory.sample_transition_batch(BATCH_SIZE)
      self.assertEqual(batch[0].shape[0], BATCH_SIZE)
    # Verify we revert to default batch size.
    for i in range(1000):
      batch = memory.sample_transition_batch()
      self.assertEqual(batch[0].shape[0], 2)

    # Verify we can specify what indices to sample.
    indices = [1, 2, 3, 5, 8]
    expected_states = np.array([
        np.full(OBSERVATION_SHAPE + (1,), i, dtype=OBS_DTYPE)
        for i in indices
    ])
    expected_next_states = (expected_states + 1) % replay_capacity
    # Because the replay buffer is circular, we can exactly compute what the
    # states will be at the specified indices by doing a little mod math:
    expected_states += num_adds - replay_capacity
    expected_next_states += num_adds - replay_capacity
    # This is replicating the formula that was used above to determine what
    # transitions are terminal when adding observation (i % 4).
    expected_terminal = np.array(
        [min((x + num_adds - replay_capacity) % 4, 1) for x in indices])
    expected_extra2 = np.zeros([len(indices), 2])
    batch = memory.sample_transition_batch(
        batch_size=len(indices), indices=indices)
    (states, action, reward, next_states, next_action, next_reward, terminal,
     indices_batch, extra1, extra2) = batch
    self.assertAllEqual(states, expected_states)
    self.assertAllEqual(action, np.zeros(len(indices)))
    self.assertAllEqual(reward, np.zeros(len(indices)))
    self.assertAllEqual(next_action, np.zeros(len(indices)))
    self.assertAllEqual(next_reward, np.zeros(len(indices)))
    self.assertAllEqual(next_states, expected_next_states)
    self.assertAllEqual(terminal, expected_terminal)
    self.assertAllEqual(indices_batch, indices)
    self.assertAllEqual(extra1, np.zeros(len(indices)))
    self.assertAllEqual(extra2, expected_extra2)

  def testSamplingWithterminalInTrajectory(self):
    replay_capacity = 10
    update_horizon = 3
    memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=1,
        replay_capacity=replay_capacity,
        batch_size=2,
        update_horizon=update_horizon,
        gamma=1.0)
    for i in range(replay_capacity):
      memory.add(
          np.full(OBSERVATION_SHAPE, i, dtype=OBS_DTYPE),
          i * 2,  # action
          i,  # reward
          1 if i == 3 else 0)  # terminal
    indices = [2, 3, 4]
    batch = memory.sample_transition_batch(batch_size=len(indices),
                                           indices=indices)
    states, action, reward, _, _, _, terminal, indices_batch = batch
    expected_states = np.array([
        np.full(OBSERVATION_SHAPE + (1,), i, dtype=OBS_DTYPE)
        for i in indices
    ])
    # The reward in the replay buffer will be (an asterisk marks the terminal
    # state):
    #   [0 1 2 3* 4 5 6 7 8 9]
    # Since we're setting the update_horizon to 3, the accumulated trajectory
    # reward starting at each of the replay buffer positions will be:
    #   [3 6 5 3 15 18 21 24]
    # Since indices = [2, 3, 4], our expected reward are [5, 3, 15].
    expected_reward = np.array([5, 3, 15])
    # Because update_horizon = 3, both indices 2 and 3 include terminal.
    expected_terminal = np.array([1, 1, 0])
    self.assertAllEqual(states, expected_states)
    self.assertAllEqual(action, np.array(indices) * 2)
    self.assertAllEqual(reward, expected_reward)
    self.assertAllEqual(terminal, expected_terminal)
    self.assertAllEqual(indices_batch, indices)

  def testInvalidRange(self):
    # The correct range accounts for the automatically applied padding (3 blanks
    # each episode.

    invalid_range = circular_replay_buffer.invalid_range(
        cursor=6, replay_capacity=10, stack_size=4, update_horizon=1)
    correct_invalid_range = [5, 6, 7, 8, 9]
    self.assertAllClose(correct_invalid_range, invalid_range)

    invalid_range = circular_replay_buffer.invalid_range(
        cursor=9, replay_capacity=10, stack_size=4, update_horizon=1)
    correct_invalid_range = [8, 9, 0, 1, 2]
    self.assertAllClose(correct_invalid_range, invalid_range)

    invalid_range = circular_replay_buffer.invalid_range(
        cursor=0, replay_capacity=10, stack_size=4, update_horizon=1)
    correct_invalid_range = [9, 0, 1, 2, 3]
    self.assertAllClose(correct_invalid_range, invalid_range)

    invalid_range = circular_replay_buffer.invalid_range(
        cursor=6, replay_capacity=10, stack_size=4, update_horizon=3)
    correct_invalid_range = [3, 4, 5, 6, 7, 8, 9]
    self.assertAllClose(correct_invalid_range, invalid_range)

  def testIsTransitionValid(self):
    memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=10,
        batch_size=2)

    memory.add(
        np.full(OBSERVATION_SHAPE, 0, dtype=OBS_DTYPE), 0, 0, 0)
    memory.add(
        np.full(OBSERVATION_SHAPE, 0, dtype=OBS_DTYPE), 0, 0, 0)
    memory.add(
        np.full(OBSERVATION_SHAPE, 0, dtype=OBS_DTYPE), 0, 0, 1)

    # These valids account for the automatically applied padding (3 blanks each
    # episode.
    correct_valids = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    # The cursor is:                    ^\
    for i in range(10):
      self.assertEqual(correct_valids[i], memory.is_valid_transition(i),
                       'Index %i should be %s' % (i, bool(correct_valids[i])))

  def testSave(self):
    memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=5,
        batch_size=BATCH_SIZE)
    memory.observation = self._test_observation
    memory.action = self._test_action
    memory.reward = self._test_reward
    memory.terminal = self._test_terminal
    current_iteration = 5
    stale_iteration = current_iteration - memory._checkpoint_duration
    memory.save(self._test_subdir, stale_iteration)
    for attr in memory.__dict__:
      if attr.startswith('_'):
        continue
      stale_filename = os.path.join(self._test_subdir, '{}_ckpt.{}.gz'.format(
          attr, stale_iteration))
      self.assertTrue(tf.io.gfile.exists(stale_filename))

    memory.save(self._test_subdir, current_iteration)
    for attr in memory.__dict__:
      if attr.startswith('_'):
        continue
      filename = os.path.join(self._test_subdir, '{}_ckpt.{}.gz'.format(
          attr, current_iteration))
      self.assertTrue(tf.io.gfile.exists(filename))
      # The stale version file should have been deleted.
      self.assertFalse(tf.io.gfile.exists(stale_filename))

  def testEpisodeEndIndicesAreCorrectlySaved(self):
    testdir = self.create_tempdir()

    memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=5,
        batch_size=BATCH_SIZE)
    memory.add(np.zeros(OBSERVATION_SHAPE, dtype=np.float32),
               0,
               0.0,
               False,
               episode_end=True)
    current_iteration = 5

    memory.save(testdir.full_path, current_iteration)

    filename = os.path.join(testdir.full_path,
                            f'episode_end_indices_ckpt.{current_iteration}.gz')
    self.assertTrue(tf.io.gfile.exists(filename))

  def testEpisodeEndIndicesAreCorrectlyLoaded(self):
    testdir = self.create_tempdir()

    memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=5,
        batch_size=BATCH_SIZE)
    memory.add(np.zeros(OBSERVATION_SHAPE, dtype=np.float32),
               0,
               0.0,
               False,
               episode_end=True)
    new_memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=5,
        batch_size=BATCH_SIZE)
    current_iteration = 5

    memory.save(testdir.full_path, current_iteration)
    new_memory.load(testdir.full_path, str(current_iteration))

    self.assertLen(new_memory.episode_end_indices, 1)
    self.assertEqual(
        memory.episode_end_indices,
        new_memory.episode_end_indices)

  def testSaveWithKeepEvery(self):
    checkpoint_duration, keep_every = 1, 2
    memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=5,
        batch_size=BATCH_SIZE,
        keep_every=keep_every,
        checkpoint_duration=checkpoint_duration)
    memory.observation = self._test_observation
    memory.action = self._test_action
    memory.reward = self._test_reward
    memory.terminal = self._test_terminal
    total_iterations = 6
    for iteration in range(1, total_iterations+1):
      memory.save(self._test_subdir, iteration)
    stale_iteration = total_iterations - memory._checkpoint_duration

    for iteration in range(1, total_iterations+1):
      for attr in memory.__dict__:
        if attr.startswith('_'):
          continue
        filename = os.path.join(self._test_subdir, '{}_ckpt.{}.gz'.format(
            attr, iteration))
        # Stale file should have been deleted if not a multiple of `keep_every`.
        if (iteration <= stale_iteration) and (iteration % keep_every) != 0:
          self.assertFalse(tf.io.gfile.exists(filename))
        else:
          self.assertTrue(tf.io.gfile.exists(filename))

  def testSaveNonNDArrayAttributes(self):
    """Tests checkpointing an attribute which is not a numpy array."""
    memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=5,
        batch_size=BATCH_SIZE)

    # Add some non-numpy data: an int, a string, an object.
    memory.dummy_attribute_1 = 4753849
    memory.dummy_attribute_2 = 'String data'
    memory.dummy_attribute_3 = CheckpointableClass()

    current_iteration = 5
    stale_iteration = current_iteration - memory._checkpoint_duration
    memory.save(self._test_subdir, stale_iteration)
    for attr in memory.__dict__:
      if attr.startswith('_'):
        continue
      stale_filename = os.path.join(self._test_subdir, '{}_ckpt.{}.gz'.format(
          attr, stale_iteration))
      self.assertTrue(tf.io.gfile.exists(stale_filename))

    memory.save(self._test_subdir, current_iteration)
    for attr in memory.__dict__:
      if attr.startswith('_'):
        continue
      filename = os.path.join(self._test_subdir, '{}_ckpt.{}.gz'.format(
          attr, current_iteration))
      self.assertTrue(tf.io.gfile.exists(filename))
      # The stale version file should have been deleted.
      self.assertFalse(tf.io.gfile.exists(stale_filename))

  def testLoadFromNonexistentDirectory(self):
    memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=5,
        batch_size=BATCH_SIZE)
    # We are trying to load from a non-existent directory, so a NotFoundError
    # will be raised.
    with self.assertRaises(tf.errors.NotFoundError):
      memory.load('/does/not/exist', '3')
    self.assertNotEqual(memory._store['observation'], self._test_observation)
    self.assertNotEqual(memory._store['action'], self._test_action)
    self.assertNotEqual(memory._store['reward'], self._test_reward)
    self.assertNotEqual(memory._store['terminal'], self._test_terminal)
    self.assertNotEqual(memory.add_count, self._test_add_count)
    self.assertNotEqual(memory.invalid_range, self._test_invalid_range)

  def testPartialLoadFails(self):
    memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=5,
        batch_size=BATCH_SIZE)
    self.assertNotEqual(memory._store['observation'], self._test_observation)
    self.assertNotEqual(memory._store['action'], self._test_action)
    self.assertNotEqual(memory._store['reward'], self._test_reward)
    self.assertNotEqual(memory._store['terminal'], self._test_terminal)
    self.assertNotEqual(memory.add_count, self._test_add_count)
    numpy_arrays = {
        'observation': self._test_observation,
        'action': self._test_action,
        'terminal': self._test_terminal,
        'add_count': self._test_add_count,
        'invalid_range': self._test_invalid_range
    }
    for attr in numpy_arrays:
      filename = os.path.join(self._test_subdir, '{}_ckpt.3.gz'.format(attr))
      with tf.io.gfile.GFile(filename, 'w') as f:
        with gzip.GzipFile(fileobj=f) as outfile:
          np.save(outfile, numpy_arrays[attr], allow_pickle=False)
    # We are are missing the reward file, so a NotFoundError will be raised.
    with self.assertRaises(tf.errors.NotFoundError):
      memory.load(self._test_subdir, '3')
    # Since we are missing the reward file, it should not have loaded any of
    # the other files.
    self.assertNotEqual(memory._store['observation'], self._test_observation)
    self.assertNotEqual(memory._store['action'], self._test_action)
    self.assertNotEqual(memory._store['reward'], self._test_reward)
    self.assertNotEqual(memory._store['terminal'], self._test_terminal)
    self.assertNotEqual(memory.add_count, self._test_add_count)
    self.assertNotEqual(memory.invalid_range, self._test_invalid_range)

  def testLoad(self):
    memory = circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=5,
        batch_size=BATCH_SIZE)
    self.assertNotEqual(memory._store['observation'], self._test_observation)
    self.assertNotEqual(memory._store['action'], self._test_action)
    self.assertNotEqual(memory._store['reward'], self._test_reward)
    self.assertNotEqual(memory._store['terminal'], self._test_terminal)
    self.assertNotEqual(memory.add_count, self._test_add_count)
    self.assertNotEqual(memory.invalid_range, self._test_invalid_range)
    store_prefix = '$store$_'
    numpy_arrays = {
        store_prefix + 'observation': self._test_observation,
        store_prefix + 'action': self._test_action,
        store_prefix + 'reward': self._test_reward,
        store_prefix + 'terminal': self._test_terminal,
        'add_count': self._test_add_count,
        'invalid_range': self._test_invalid_range
    }
    for attr in numpy_arrays:
      filename = os.path.join(self._test_subdir, '{}_ckpt.3.gz'.format(attr))
      with tf.io.gfile.GFile(filename, 'w') as f:
        with gzip.GzipFile(fileobj=f) as outfile:
          np.save(outfile, numpy_arrays[attr], allow_pickle=False)
    memory.load(self._test_subdir, '3')
    self.assertAllClose(memory._store['observation'], self._test_observation)
    self.assertAllClose(memory._store['action'], self._test_action)
    self.assertAllClose(memory._store['reward'], self._test_reward)
    self.assertAllClose(memory._store['terminal'], self._test_terminal)
    self.assertEqual(memory.add_count, self._test_add_count)
    self.assertAllClose(memory.invalid_range, self._test_invalid_range)


class WrappedReplayBufferTest(tf.test.TestCase):

  def setUp(self):
    super(WrappedReplayBufferTest, self).setUp()
    self._test_subdir = os.path.join('/tmp/dopamine_tests', 'wrapped_replay')
    shutil.rmtree(self._test_subdir, ignore_errors=True)
    os.makedirs(self._test_subdir)
    num_dims = 10
    self._test_observation = np.ones(num_dims) * 3
    self._test_action = np.ones(num_dims) * 5
    self._test_reward = np.ones(num_dims) * 7
    self._test_terminal = np.ones(num_dims) * 11
    self._test_add_count = np.array(7)
    self._test_invalid_range = np.ones(num_dims)

  def testConstructorCapacityNotLargeEnough(self):
    with self.assertRaisesRegex(
        ValueError,
        r'Update horizon \(5\) should be significantly '
        r'smaller than replay capacity \(5\)\.'):
      circular_replay_buffer.WrappedReplayBuffer(
          observation_shape=OBSERVATION_SHAPE,
          stack_size=STACK_SIZE,
          replay_capacity=5,
          update_horizon=5)

  def testConstructorWithZeroUpdateHorizon(self):
    with self.assertRaisesRegex(ValueError,
                                r'Update horizon must be positive\.'):
      circular_replay_buffer.WrappedReplayBuffer(
          observation_shape=OBSERVATION_SHAPE,
          stack_size=STACK_SIZE,
          update_horizon=0)

  def testConstructorWithOutOfBoundsDiscountFactor(self):
    exception_string = r'Discount factor \(gamma\) must be in \[0, 1\]\.'
    with self.assertRaisesRegex(ValueError, exception_string):
      circular_replay_buffer.WrappedReplayBuffer(
          observation_shape=OBSERVATION_SHAPE, stack_size=STACK_SIZE, gamma=-1)
    with self.assertRaisesRegex(ValueError, exception_string):
      circular_replay_buffer.WrappedReplayBuffer(
          observation_shape=OBSERVATION_SHAPE, stack_size=STACK_SIZE, gamma=1.1)

  def testConstructorWithExtraStorageTypes(self):
    circular_replay_buffer.WrappedReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        extra_storage_types=[
            circular_replay_buffer.ReplayElement('extra1', [], np.float32),
            circular_replay_buffer.ReplayElement('extra2', [2], np.int8)
        ])

  def _verify_sampled_trajectories(self, batch):
    (states, action, reward, next_states, next_action, next_reward, terminal,
     indices) = batch.values()
    # Because we've added BATCH_SIZE * 2 observation, using the enumerator to
    # fill the respective observation, all observation will be np.full arrays
    # where the values are in [0, BATCH_SIZE * 2). Because we're sampling we can
    # deterministically predict what values will be sampled, we can ensure that
    # they will all be in that range by setting a "midpoint" observation (with
    # value BATCH_SIZE), and verifying that all observation values are near this
    # midpoint, with tolerance BATCH_SIZE.
    midpoint_observation = np.full(
        (BATCH_SIZE,) + OBSERVATION_SHAPE + (STACK_SIZE,),
        BATCH_SIZE,
        dtype=OBS_DTYPE)
    self.assertAllClose(states, midpoint_observation, rtol=BATCH_SIZE)
    self.assertAllClose(next_states, midpoint_observation, rtol=BATCH_SIZE)
    self.assertAllClose(action, np.ones(BATCH_SIZE) * 2)
    self.assertAllClose(reward, np.ones(BATCH_SIZE))
    self.assertAllClose(next_action, np.ones(BATCH_SIZE) * 2)
    self.assertAllClose(next_reward, np.ones(BATCH_SIZE))
    self.assertAllClose(terminal, np.zeros(BATCH_SIZE))
    self.assertAllClose(indices, np.ones(BATCH_SIZE) * BATCH_SIZE,
                        rtol=BATCH_SIZE)

  def testConstructorWithNoStaging(self):
    replay = circular_replay_buffer.WrappedReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=100,
        batch_size=BATCH_SIZE,
        use_staging=False)
    with self.test_session() as sess:
      for i in range(BATCH_SIZE * 2):
        observation = np.full(OBSERVATION_SHAPE, i, dtype=OBS_DTYPE)
        replay.add(observation, 2, 1, 0)
    self._verify_sampled_trajectories(sess.run(replay.transition))

  def testConstructorWithStaging(self):
    replay = circular_replay_buffer.WrappedReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=100,
        batch_size=BATCH_SIZE,
        use_staging=True)
    with self.test_session() as sess:
      for i in range(BATCH_SIZE * 2):
        observation = np.full(OBSERVATION_SHAPE, i, dtype=OBS_DTYPE)
        replay.add(observation, 2, 1, 0)
    self._verify_sampled_trajectories(sess.run(replay.transition))

  def testWrapperSave(self):
    replay = circular_replay_buffer.WrappedReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=5,
        batch_size=BATCH_SIZE)
    replay.memory.observation = self._test_observation
    replay.memory.action = self._test_action
    replay.memory.reward = self._test_reward
    replay.memory.terminal = self._test_terminal
    replay.memory.add_count = self._test_add_count
    replay.memory.invalid_range = self._test_invalid_range
    replay.save(self._test_subdir, 3)
    for attr in replay.memory.__dict__:
      if attr.startswith('_'):
        continue
      filename = os.path.join(self._test_subdir, '{}_ckpt.3.gz'.format(attr))
      self.assertTrue(tf.io.gfile.exists(filename))

  def testWrapperLoad(self):
    replay = circular_replay_buffer.WrappedReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=5,
        batch_size=BATCH_SIZE)
    self.assertNotEqual(replay.memory._store['observation'],
                        self._test_observation)
    self.assertNotEqual(replay.memory._store['action'], self._test_action)
    self.assertNotEqual(replay.memory._store['reward'], self._test_reward)
    self.assertNotEqual(replay.memory._store['terminal'], self._test_terminal)
    self.assertNotEqual(replay.memory.add_count, self._test_add_count)
    self.assertNotEqual(replay.memory.invalid_range, self._test_invalid_range)
    store_prefix = '$store$_'
    numpy_arrays = {
        store_prefix + 'observation': self._test_observation,
        store_prefix + 'action': self._test_action,
        store_prefix + 'reward': self._test_reward,
        store_prefix + 'terminal': self._test_terminal,
        'add_count': self._test_add_count,
        'invalid_range': self._test_invalid_range
    }
    for attr in numpy_arrays:
      filename = os.path.join(self._test_subdir, '{}_ckpt.3.gz'.format(attr))
      with tf.io.gfile.GFile(filename, 'w') as f:
        with gzip.GzipFile(fileobj=f) as outfile:
          np.save(outfile, numpy_arrays[attr], allow_pickle=False)
    replay.load(self._test_subdir, '3')
    self.assertAllClose(replay.memory._store['observation'],
                        self._test_observation)
    self.assertAllClose(replay.memory._store['action'], self._test_action)
    self.assertAllClose(replay.memory._store['reward'], self._test_reward)
    self.assertAllClose(replay.memory._store['terminal'], self._test_terminal)
    self.assertEqual(replay.memory.add_count, self._test_add_count)
    self.assertAllClose(replay.memory.invalid_range, self._test_invalid_range)

  def testDefaultObsDataType(self):
    # Tests that the default data type for observations is np.uint8 for
    # integration with Atari 2600.
    replay = circular_replay_buffer.WrappedReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE, replay_capacity=10)
    self.assertEqual(replay.memory._store['observation'].dtype, np.uint8)

  def testCustomObsDataType(self):
    # Tests that observation store is initialized with the correct data type
    # when an observation_dtype argument is passed to the constructor.
    replay = circular_replay_buffer.WrappedReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE, replay_capacity=10, observation_dtype=np.int32)
    self.assertEqual(replay.memory._store['observation'].dtype, np.int32)


if __name__ == '__main__':
  tf.compat.v1.disable_v2_behavior()
  tf.test.main()

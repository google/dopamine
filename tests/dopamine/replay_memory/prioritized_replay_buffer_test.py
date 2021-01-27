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
"""Tests for prioritzed replay memory."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from dopamine.replay_memory import circular_replay_buffer
from dopamine.replay_memory import prioritized_replay_buffer
import numpy as np
import tensorflow as tf


# Default parameters used when creating the replay memory.
SCREEN_SIZE = (84, 84)
STACK_SIZE = 4
BATCH_SIZE = 32
REPLAY_CAPACITY = 100


class OutOfGraphPrioritizedReplayBufferTest(tf.test.TestCase):

  def create_default_memory(self, extra_storage_types=None):
    return prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer(
        SCREEN_SIZE,
        STACK_SIZE,
        REPLAY_CAPACITY,
        BATCH_SIZE,
        extra_storage_types=extra_storage_types,
        max_sample_attempts=10)  # For faster tests.

  def add_blank(self, memory, action=0, reward=0.0, terminal=0, priority=1.0):
    """Adds a replay transition with a blank observation.

    Allows setting action, reward, terminal.
    Args:
      memory: The replay memory.
      action: Integer.
      reward: Float.
      terminal: Integer (0 or 1).
      priority: Float. Defults to standard priority of 1.

    Returns:
      Index of the transition just added.
    """
    dummy = np.zeros(SCREEN_SIZE)
    memory.add(dummy, action, reward, terminal, priority=priority)
    index = (memory.cursor() - 1) % REPLAY_CAPACITY
    return index

  def testAddWithAndWithoutPriority(self):
    memory = self.create_default_memory()
    self.assertEqual(memory.cursor(), 0)
    zeros = np.zeros(SCREEN_SIZE)

    self.add_blank(memory)
    self.assertEqual(memory.cursor(), STACK_SIZE)
    self.assertEqual(memory.add_count, STACK_SIZE)

    # Check that the prioritized replay buffer expects an additional argument
    # for priority.
    with self.assertRaisesRegexp(ValueError, 'Add expects'):
      memory.add(zeros, 0, 0, 0)

  def testAddWithAdditionalArgsAndPriority(self):
    memory = self.create_default_memory(extra_storage_types=[
        circular_replay_buffer.ReplayElement('test_item', (), np.float32)
    ])
    self.assertEqual(memory.cursor(), 0)
    zeros = np.zeros(SCREEN_SIZE)

    memory.add(zeros, 0, 0.0, 0, 0.0, priority=1.0)
    self.assertEqual(memory.cursor(), STACK_SIZE)
    self.assertEqual(memory.add_count, STACK_SIZE)

    # Check that the prioritized replay buffer expects an additional argument
    # for test_item.
    with self.assertRaisesRegexp(ValueError, 'Add expects'):
      memory.add(zeros, 0, 0, 0, priority=1.0)

  def testDummyScreensAddedToNewMemory(self):
    memory = self.create_default_memory()
    index = self.add_blank(memory)
    for i in range(index):
      self.assertEqual(memory.sum_tree.get(i), 0.0)

  def testGetPriorityWithInvalidIndices(self):
    memory = self.create_default_memory()
    index = self.add_blank(memory)
    with self.assertRaises(AssertionError, msg='Indices must be an array.'):
      memory.get_priority(index)
    with self.assertRaises(AssertionError,
                           msg='Indices must be int32s, given: int64'):
      memory.get_priority(np.array([index]))

  def testSetAndGetPriority(self):
    memory = self.create_default_memory()
    batch_size = 7
    indices = np.zeros(batch_size, dtype=np.int32)
    for index in range(batch_size):
      indices[index] = self.add_blank(memory)
    priorities = np.arange(batch_size)
    memory.set_priority(indices, priorities)
    # We send the indices in reverse order and verify the priorities come back
    # in that same order.
    fetched_priorities = memory.get_priority(np.flip(indices, 0))
    for i in range(batch_size):
      self.assertEqual(priorities[i], fetched_priorities[batch_size - 1 - i])

  def testNewElementHasHighPriority(self):
    memory = self.create_default_memory()
    index = self.add_blank(memory)
    self.assertEqual(
        memory.get_priority(np.array([index], dtype=np.int32))[0],
        1.0)

  def testLowPriorityElementNotFrequentlySampled(self):
    memory = self.create_default_memory()
    # Add an item and set its priority to 0.
    self.add_blank(memory, terminal=0, priority=0.0)
    # Now add a few new items.
    for _ in range(3):
      self.add_blank(memory, terminal=1)
    # This test should always pass.
    for _ in range(100):
      _, _, _, _, _, _, terminals, _, _ = (
          memory.sample_transition_batch(batch_size=2))
      # Ensure all terminals are set to 1.
      self.assertTrue((terminals == 1).all())

  def testSampleIndexBatchTooManyFailedRetries(self):
    memory = self.create_default_memory()
    # Only adding a single observation is not enough to be able to sample
    # (as it both straddles the cursor and does not pass the
    # `index >= self.cursor() - self._update_horizon` check in
    # circular_replay_buffer.py).
    self.add_blank(memory)
    with self.assertRaises(
        RuntimeError,
        msg='Max sample attempts: Tried 10 times but only sampled 1 valid '
            'indices. Batch size is 2'):
      memory.sample_index_batch(2)

  def testSampleIndexBatch(self):
    memory = prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer(
        SCREEN_SIZE,
        STACK_SIZE,
        REPLAY_CAPACITY,
        BATCH_SIZE,
        max_sample_attempts=REPLAY_CAPACITY)
    # This will ensure we end up with cursor == 1.
    for _ in range(REPLAY_CAPACITY - STACK_SIZE + 2):
      self.add_blank(memory)
    self.assertEqual(memory.cursor(), 1)
    samples = memory.sample_index_batch(REPLAY_CAPACITY)
    # Because cursor == 1, the invalid range as set by circular_replay_buffer.py
    # will be # [0, 1, 2, 3], resulting in all samples being in
    # [STACK_SIZE, REPLAY_CAPACITY - 1].
    for sample in samples:
      self.assertGreaterEqual(sample, STACK_SIZE)
      self.assertLessEqual(sample, REPLAY_CAPACITY - 1)


class WrappedPrioritizedReplayBufferTest(tf.test.TestCase):
  """Tests the Tensorflow wrapper around the Python replay memory.
  """

  def create_default_memory(self):
    return prioritized_replay_buffer.WrappedPrioritizedReplayBuffer(
        SCREEN_SIZE,
        STACK_SIZE,
        use_staging=False,
        replay_capacity=REPLAY_CAPACITY,
        batch_size=BATCH_SIZE,
        max_sample_attempts=10)  # For faster tests.

  def add_blank(self, replay):
    replay.add(np.zeros(SCREEN_SIZE), 0, 0, 0, 1.0)

  def testSetAndGetPriority(self):
    replay = self.create_default_memory()

    batch_size = 7
    with self.test_session() as sess:
      indices = np.zeros(batch_size, dtype=np.int32)
      for index in range(batch_size):
        self.add_blank(replay)
        indices[index] = replay.memory.cursor() - 1

      priorities = np.arange(batch_size)
      sess.run(replay.tf_set_priority(indices, priorities))
      # We send the indices in reverse order and verify the priorities come back
      # in that same order.
      fetched_priorities = sess.run(replay.tf_get_priority(np.flip(indices, 0)))
      for i in range(batch_size):
        self.assertEqual(priorities[i],
                         fetched_priorities[batch_size - 1 - i])

  def testSampleBatch(self):
    replay = self.create_default_memory()

    num_data = 64
    with self.test_session() as sess:
      for _ in range(num_data):
        self.add_blank(replay)
      probabilities = sess.run(replay.transition['sampling_probabilities'])
      for prob in probabilities:
        self.assertEqual(prob, 1.0)

  def testConstructorWithExtraStorageTypes(self):
    prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer(
        SCREEN_SIZE,
        STACK_SIZE,
        REPLAY_CAPACITY,
        BATCH_SIZE,
        extra_storage_types=[
            prioritized_replay_buffer.ReplayElement('extra1', [], np.float32),
            prioritized_replay_buffer.ReplayElement('extra2', [2], np.int8)
        ])

if __name__ == '__main__':
  tf.compat.v1.disable_v2_behavior()
  tf.test.main()

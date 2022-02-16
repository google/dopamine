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
"""Tests for dopamine.common.checkpointer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os



from absl import flags
from dopamine.discrete_domains import checkpointer
import tensorflow as tf

FLAGS = flags.FLAGS


class CheckpointerTest(tf.test.TestCase):

  def testCheckpointingInitialization(self):
    # Fails with empty directory.
    with self.assertRaisesRegex(ValueError,
                                'No path provided to Checkpointer.'):
      checkpointer.Checkpointer('')
    # Fails with invalid directory.
    invalid_dir = '/does/not/exist'
    with self.assertRaisesRegex(
        ValueError, 'Unable to create checkpoint path: {}.'.
        format(invalid_dir)):
      checkpointer.Checkpointer(invalid_dir)
    # Succeeds with valid directory.
    tmpdir = self.create_tempdir()
    checkpointer.Checkpointer(tmpdir)
    # This verifies initialization still works after the directory has already
    # been created.
    self.assertTrue(tf.io.gfile.exists(tmpdir))
    checkpointer.Checkpointer(tmpdir)

  def testLogToFileWithValidDirectoryDefaultPrefix(self):
    tmpdir = self.create_tempdir()
    exp_checkpointer = checkpointer.Checkpointer(tmpdir)
    data = {'data1': 1, 'data2': 'two', 'data3': (3, 'three')}
    iteration_number = 1729
    exp_checkpointer.save_checkpoint(iteration_number, data)
    loaded_data = exp_checkpointer.load_checkpoint(iteration_number)
    self.assertEqual(data, loaded_data)
    self.assertIsNone(exp_checkpointer.load_checkpoint(iteration_number + 1))

  def testLogToFileWithValidDirectoryCustomPrefix(self):
    prefix = 'custom_prefix'
    tmpdir = self.create_tempdir()
    exp_checkpointer = checkpointer.Checkpointer(
        tmpdir,
        checkpoint_file_prefix=prefix)
    data = {'data1': 1, 'data2': 'two', 'data3': (3, 'three')}
    iteration_number = 1729
    exp_checkpointer.save_checkpoint(iteration_number, data)
    loaded_data = exp_checkpointer.load_checkpoint(iteration_number)
    self.assertEqual(data, loaded_data)
    self.assertIsNone(exp_checkpointer.load_checkpoint(iteration_number + 1))

  def testLoadLatestCheckpointWithInvalidDir(self):
    self.assertEqual(
        -1, checkpointer.get_latest_checkpoint_number('/does/not/exist'))

  def testLoadLatestCheckpointWithEmptyDir(self):
    tmpdir = self.create_tempdir()
    self.assertEqual(-1, checkpointer.get_latest_checkpoint_number(tmpdir))

  def testLoadLatestCheckpointWithOverride(self):
    override_number = 1729
    self.assertEqual(
        override_number,
        checkpointer.get_latest_checkpoint_number(
            '/ignored', override_number=override_number))

  def testLoadLatestCheckpoint(self):
    tmpdir = self.create_tempdir()
    exp_checkpointer = checkpointer.Checkpointer(tmpdir)
    first_iter = 1729
    exp_checkpointer.save_checkpoint(first_iter, first_iter)
    second_iter = first_iter + 1
    exp_checkpointer.save_checkpoint(second_iter, second_iter)
    self.assertEqual(
        second_iter,
        checkpointer.get_latest_checkpoint_number(tmpdir))

  def testGarbageCollection(self):
    custom_prefix = 'custom_prefix'
    tmpdir = self.create_tempdir()
    exp_checkpointer = checkpointer.Checkpointer(
        tmpdir,
        checkpoint_file_prefix=custom_prefix)
    data = {'data1': 1, 'data2': 'two', 'data3': (3, 'three')}
    deleted_log_files = 7
    total_log_files = exp_checkpointer._checkpoint_duration + deleted_log_files
    for iteration_number in range(total_log_files):
      exp_checkpointer.save_checkpoint(iteration_number, data)
    for iteration_number in range(total_log_files):
      prefixes = [custom_prefix, 'sentinel_checkpoint_complete']
      for prefix in prefixes:
        checkpoint_file = os.path.join(tmpdir,
                                       f'{prefix}.{iteration_number}')
        if iteration_number < deleted_log_files:
          self.assertFalse(tf.io.gfile.exists(checkpoint_file))
        else:
          self.assertTrue(tf.io.gfile.exists(checkpoint_file))

  def testGarbageCollectionWithCheckpointFrequency(self):
    custom_prefix = 'custom_prefix'
    checkpoint_frequency = 3
    tmpdir = self.create_tempdir()
    exp_checkpointer = checkpointer.Checkpointer(
        tmpdir,
        checkpoint_file_prefix=custom_prefix,
        checkpoint_frequency=checkpoint_frequency)
    data = {'data1': 1, 'data2': 'two', 'data3': (3, 'three')}
    deleted_log_files = 6
    total_log_files = (exp_checkpointer._checkpoint_duration *
                       checkpoint_frequency) + deleted_log_files + 1

    # The checkpoints will happen in iteration numbers 0,3,6,9,12,15,18.
    # We are checking if checkpoints 0,3,6 are deleted.
    for iteration_number in range(total_log_files):
      exp_checkpointer.save_checkpoint(iteration_number,
                                       data)
    for iteration_number in range(total_log_files):
      prefixes = [custom_prefix, 'sentinel_checkpoint_complete']
      for prefix in prefixes:
        checkpoint_file = os.path.join(tmpdir,
                                       f'{prefix}.{iteration_number}')
        if iteration_number <= deleted_log_files:
          self.assertFalse(tf.io.gfile.exists(checkpoint_file))
        elif iteration_number % checkpoint_frequency == 0:
          self.assertTrue(tf.io.gfile.exists(checkpoint_file))
        else:
          self.assertFalse(tf.io.gfile.exists(checkpoint_file))

  def testGarbageCollectionWithCheckpointDuration(self):
    custom_prefix = 'custom_prefix'
    checkpoint_frequency = 3
    checkpoint_duration = 6
    tmpdir = self.create_tempdir()
    exp_checkpointer = checkpointer.Checkpointer(
        tmpdir,
        checkpoint_file_prefix=custom_prefix,
        checkpoint_frequency=checkpoint_frequency,
        checkpoint_duration=checkpoint_duration)
    data = {'data1': 1, 'data2': 'two', 'data3': (3, 'three')}
    total_log_files = 40
    deleted_log_files = 21

    # The checkpoints will happen in iteration numbers
    # 0,3,6,9,12,15,18,21,24,27,30,33,36,39
    # We are checking if checkpoints 0,3,6,9,12,15,18,21 are deleted.
    for iteration_number in range(total_log_files):
      exp_checkpointer.save_checkpoint(iteration_number,
                                       data)
    for iteration_number in range(total_log_files):
      prefixes = [custom_prefix, 'sentinel_checkpoint_complete']
      for prefix in prefixes:
        checkpoint_file = os.path.join(tmpdir,
                                       f'{prefix}.{iteration_number}')
        if iteration_number <= deleted_log_files:
          self.assertFalse(tf.io.gfile.exists(checkpoint_file))
        elif iteration_number % checkpoint_frequency == 0:
          self.assertTrue(tf.io.gfile.exists(checkpoint_file))
        else:
          self.assertFalse(tf.io.gfile.exists(checkpoint_file))

  def testGarbageCollectionWithKeepEvery(self):
    custom_prefix = 'custom_prefix'
    checkpoint_frequency = 3
    checkpoint_duration = 6
    keep_every = 3
    tmpdir = self.create_tempdir()
    exp_checkpointer = checkpointer.Checkpointer(
        tmpdir,
        checkpoint_file_prefix=custom_prefix,
        checkpoint_frequency=checkpoint_frequency,
        checkpoint_duration=checkpoint_duration,
        keep_every=keep_every)
    data = {'data1': 1, 'data2': 'two', 'data3': (3, 'three')}
    deleted_log_files = 21
    total_log_files = 40

    # The checkpoints will happen in iteration numbers
    # 0,3,6,9,12,15,18,21,24,27,30,33,36,39
    # We are checking if checkpoints 3,6,12,15 are deleted.
    # Checkpoints 0, 9 and 18 would have normally been deleted but should
    # have been spared by keep_every.
    for iteration_number in range(total_log_files):
      exp_checkpointer.save_checkpoint(iteration_number,
                                       data)
    for iteration_number in range(total_log_files):
      prefixes = [custom_prefix, 'sentinel_checkpoint_complete']
      for prefix in prefixes:
        checkpoint_file = os.path.join(tmpdir,
                                       f'{prefix}.{iteration_number}')
        if (iteration_number <= deleted_log_files
            and iteration_number % (checkpoint_frequency*keep_every) != 0):
          self.assertFalse(tf.io.gfile.exists(checkpoint_file))
        elif iteration_number % checkpoint_frequency == 0:
          self.assertTrue(tf.io.gfile.exists(checkpoint_file))
        else:
          self.assertFalse(tf.io.gfile.exists(checkpoint_file))

if __name__ == '__main__':
  tf.compat.v1.disable_v2_behavior()
  tf.test.main()

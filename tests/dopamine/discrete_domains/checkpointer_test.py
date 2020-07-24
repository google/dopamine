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
import shutil



from absl import flags
from dopamine.discrete_domains import checkpointer
import tensorflow as tf

FLAGS = flags.FLAGS


class CheckpointerTest(tf.test.TestCase):

  def setUp(self):
    super(CheckpointerTest, self).setUp()
    self._test_subdir = os.path.join('/tmp/dopamine_tests', 'checkpointing')
    shutil.rmtree(self._test_subdir, ignore_errors=True)
    os.makedirs(self._test_subdir)

  def testCheckpointingInitialization(self):
    # Fails with empty directory.
    with self.assertRaisesRegexp(ValueError,
                                 'No path provided to Checkpointer.'):
      checkpointer.Checkpointer('')
    # Fails with invalid directory.
    invalid_dir = '/does/not/exist'
    with self.assertRaisesRegexp(
        ValueError, 'Unable to create checkpoint path: {}.'.
        format(invalid_dir)):
      checkpointer.Checkpointer(invalid_dir)
    # Succeeds with valid directory.
    checkpointer.Checkpointer('/tmp/dopamine_tests')
    # This verifies initialization still works after the directory has already
    # been created.
    self.assertTrue(tf.io.gfile.exists('/tmp/dopamine_tests'))
    checkpointer.Checkpointer('/tmp/dopamine_tests')

  def testLogToFileWithValidDirectoryDefaultPrefix(self):
    exp_checkpointer = checkpointer.Checkpointer(self._test_subdir)
    data = {'data1': 1, 'data2': 'two', 'data3': (3, 'three')}
    iteration_number = 1729
    exp_checkpointer.save_checkpoint(iteration_number, data)
    loaded_data = exp_checkpointer.load_checkpoint(iteration_number)
    self.assertEqual(data, loaded_data)
    self.assertIsNone(exp_checkpointer.load_checkpoint(iteration_number + 1))

  def testLogToFileWithValidDirectoryCustomPrefix(self):
    prefix = 'custom_prefix'
    exp_checkpointer = checkpointer.Checkpointer(self._test_subdir,
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
    self.assertEqual(
        -1, checkpointer.get_latest_checkpoint_number(self._test_subdir))

  def testLoadLatestCheckpointWithOverride(self):
    override_number = 1729
    self.assertEqual(
        override_number,
        checkpointer.get_latest_checkpoint_number(
            '/ignored', override_number=override_number))

  def testLoadLatestCheckpoint(self):
    exp_checkpointer = checkpointer.Checkpointer(self._test_subdir)
    first_iter = 1729
    exp_checkpointer.save_checkpoint(first_iter, first_iter)
    second_iter = first_iter + 1
    exp_checkpointer.save_checkpoint(second_iter, second_iter)
    self.assertEqual(
        second_iter,
        checkpointer.get_latest_checkpoint_number(self._test_subdir))

  def testGarbageCollection(self):
    custom_prefix = 'custom_prefix'
    exp_checkpointer = checkpointer.Checkpointer(
        self._test_subdir, checkpoint_file_prefix=custom_prefix)
    data = {'data1': 1, 'data2': 'two', 'data3': (3, 'three')}
    deleted_log_files = 7
    total_log_files = checkpointer.CHECKPOINT_DURATION + deleted_log_files
    for iteration_number in range(total_log_files):
      exp_checkpointer.save_checkpoint(iteration_number, data)
    for iteration_number in range(total_log_files):
      prefixes = [custom_prefix, 'sentinel_checkpoint_complete']
      for prefix in prefixes:
        checkpoint_file = os.path.join(self._test_subdir, '{}.{}'.format(
            prefix, iteration_number))
        if iteration_number < deleted_log_files:
          self.assertFalse(tf.io.gfile.exists(checkpoint_file))
        else:
          self.assertTrue(tf.io.gfile.exists(checkpoint_file))

  def testGarbageCollectionWithCheckpointFrequency(self):
    custom_prefix = 'custom_prefix'
    checkpoint_frequency = 3
    exp_checkpointer = checkpointer.Checkpointer(
        self._test_subdir, checkpoint_file_prefix=custom_prefix,
        checkpoint_frequency=checkpoint_frequency)
    data = {'data1': 1, 'data2': 'two', 'data3': (3, 'three')}
    deleted_log_files = 6
    total_log_files = (checkpointer.CHECKPOINT_DURATION *
                       checkpoint_frequency) + deleted_log_files + 1

    # The checkpoints will happen in iteration numbers 0,3,6,9,12,15,18.
    # We are checking if checkpoints 0,3,6 are deleted.
    for iteration_number in range(total_log_files):
      exp_checkpointer.save_checkpoint(iteration_number,
                                       data)
    for iteration_number in range(total_log_files):
      prefixes = [custom_prefix, 'sentinel_checkpoint_complete']
      for prefix in prefixes:
        checkpoint_file = os.path.join(self._test_subdir, '{}.{}'.format(
            prefix, iteration_number))
        if iteration_number <= deleted_log_files:
          self.assertFalse(tf.io.gfile.exists(checkpoint_file))
        else:
          if iteration_number % checkpoint_frequency == 0:
            self.assertTrue(tf.io.gfile.exists(checkpoint_file))
          else:
            self.assertFalse(tf.io.gfile.exists(checkpoint_file))

if __name__ == '__main__':
  tf.compat.v1.disable_v2_behavior()
  tf.test.main()

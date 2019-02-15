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
"""Tests for dopamine.logger."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import shutil


from absl import flags
from dopamine.discrete_domains import logger
import tensorflow as tf

FLAGS = flags.FLAGS


class LoggerTest(tf.test.TestCase):

  def setUp(self):
    self._test_subdir = os.path.join('/tmp/dopamine_tests', 'logging')
    shutil.rmtree(self._test_subdir, ignore_errors=True)
    os.makedirs(self._test_subdir)

  def testLoggingDisabledWithEmptyDirectory(self):
    exp_logger = logger.Logger('')
    self.assertFalse(exp_logger.is_logging_enabled())

  def testLoggingDisabledWithInvalidDirectory(self):
    exp_logger = logger.Logger('/does/not/exist')
    self.assertFalse(exp_logger.is_logging_enabled())

  def testLoggingEnabledWithValidDirectory(self):
    exp_logger = logger.Logger('/tmp/dopamine_tests')
    self.assertTrue(exp_logger.is_logging_enabled())

  def testSetEntry(self):
    exp_logger = logger.Logger('/tmp/dopamine_tests')
    self.assertEqual(len(exp_logger.data.keys()), 0)
    key = 'key'
    val = [1, 2, 3, 4]
    exp_logger[key] = val
    expected_dictionary = {}
    expected_dictionary[key] = val
    self.assertEqual(expected_dictionary, exp_logger.data)
    # Calling __setitem__ with the same value should overwrite the previous
    # value.
    val = 'new value'
    exp_logger[key] = val
    expected_dictionary[key] = val
    self.assertEqual(expected_dictionary, exp_logger.data)

  def testLogToFileWithInvalidDirectory(self):
    exp_logger = logger.Logger('/does/not/exist')
    self.assertFalse(exp_logger.is_logging_enabled())
    exp_logger.log_to_file(None, None)

  def testLogToFileWithValidDirectory(self):
    exp_logger = logger.Logger(self._test_subdir)
    self.assertTrue(exp_logger.is_logging_enabled())
    key = 'key'
    val = [1, 2, 3, 4]
    exp_logger[key] = val
    expected_dictionary = {}
    expected_dictionary[key] = val
    self.assertEqual(expected_dictionary, exp_logger.data)
    iteration_number = 7
    exp_logger.log_to_file('log', iteration_number)
    log_file = os.path.join(self._test_subdir,
                            'log_{}'.format(iteration_number))
    with tf.gfile.GFile(log_file, 'rb') as f:
      contents = f.read()
    self.assertEqual(contents, pickle.dumps(expected_dictionary,
                                            protocol=pickle.HIGHEST_PROTOCOL))

  def testGarbageCollection(self):
    exp_logger = logger.Logger(self._test_subdir)
    self.assertTrue(exp_logger.is_logging_enabled())
    key = 'key'
    val = [1, 2, 3, 4]
    exp_logger[key] = val
    expected_dictionary = {}
    expected_dictionary[key] = val
    self.assertEqual(expected_dictionary, exp_logger.data)
    deleted_log_files = 7
    total_log_files = logger.CHECKPOINT_DURATION + deleted_log_files
    for iteration_number in range(total_log_files):
      exp_logger.log_to_file('log', iteration_number)
    for iteration_number in range(total_log_files):
      log_file = os.path.join(self._test_subdir,
                              'log_{}'.format(iteration_number))
      if iteration_number < deleted_log_files:
        self.assertFalse(tf.gfile.Exists(log_file))
      else:
        self.assertTrue(tf.gfile.Exists(log_file))


if __name__ == '__main__':
  tf.test.main()

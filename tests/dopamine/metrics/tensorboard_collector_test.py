# coding=utf-8
# Copyright 2022 The Dopamine Authors.
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
"""Tests for dopamine.metrics.tensorboard_collector."""

import os.path as osp
from unittest import mock

from absl.testing import absltest
from dopamine.metrics import statistics_instance
from dopamine.metrics import tensorboard_collector
import tensorflow as tf


class TensorboardCollectorTest(absltest.TestCase):

  def test_with_invalid_base_dir_raises_value_error(self):
    with self.assertRaises(ValueError):
      tensorboard_collector.TensorboardCollector(None)

  def test_valid_creation_with_all_required_parameters(self):
    tf.summary.create_file_writer = mock.MagicMock()
    base_dir = self.create_tempdir().full_path
    collector = tensorboard_collector.TensorboardCollector(base_dir)
    self.assertEqual(collector._base_dir,
                     osp.join(base_dir, 'metrics/tensorboard'))
    self.assertTrue(osp.exists(collector._base_dir))
    self.assertEqual(tf.summary.create_file_writer.call_count, 1)
    self.assertEqual(tf.summary.create_file_writer.call_args[0][0],
                     collector._base_dir)

  def test_write(self):
    tf.summary.create_file_writer = mock.MagicMock()
    collector = tensorboard_collector.TensorboardCollector(
        self.create_tempdir().full_path)
    self.assertEqual(1, tf.summary.create_file_writer.call_count)

    tf.summary.scalar = mock.MagicMock()
    collector.summary_writer.flush = mock.MagicMock()
    num_steps = 100
    for i in range(num_steps):
      stat = statistics_instance.StatisticsInstance('val', i, i)
      collector.write([stat])
      self.assertEqual('val', tf.summary.scalar.call_args_list[-1][0][0])
      self.assertEqual(i, tf.summary.scalar.call_args_list[-1][0][1])
      self.assertEqual({'step': i}, tf.summary.scalar.call_args_list[-1][1])

    self.assertEqual(tf.summary.scalar.call_count, num_steps)
    self.assertEqual(collector.summary_writer.flush.call_count, 0)

  def test_no_write_with_unsupported_type(self):
    tf.summary.create_file_writer = mock.MagicMock()
    collector = tensorboard_collector.TensorboardCollector(
        self.create_tempdir().full_path)
    self.assertEqual(1, tf.summary.create_file_writer.call_count)
    tf.summary.scalar = mock.MagicMock()
    for i in range(10):
      stat = statistics_instance.StatisticsInstance(
          'val', i, i, type='unsupported')
      collector.write([stat])
      tf.summary.scalar.assert_not_called()

  def test_full_run(self):
    tf.summary.create_file_writer = mock.MagicMock()
    collector = tensorboard_collector.TensorboardCollector(
        self.create_tempdir().full_path)
    tf.summary.scalar = mock.MagicMock()
    collector.summary_writer.flush = mock.MagicMock()
    num_iterations = 3
    total_steps = 0
    for i in range(num_iterations):
      num_steps = 10 * (i + 1)
      total_steps += num_steps
      for j in range(num_steps):
        val_str = f'val_{j % 2}'
        stat = statistics_instance.StatisticsInstance(val_str, j, i)
        collector.write([stat])
        self.assertEqual(val_str, tf.summary.scalar.call_args_list[-1][0][0])
        self.assertEqual(j, tf.summary.scalar.call_args_list[-1][0][1])
        self.assertEqual({'step': i}, tf.summary.scalar.call_args_list[-1][1])
      collector.flush()
    self.assertEqual(tf.summary.scalar.call_count, total_steps)
    self.assertEqual(collector.summary_writer.flush.call_count, num_iterations)


if __name__ == '__main__':
  absltest.main()

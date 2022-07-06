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
"""Tests for dopamine.metrics.console_collector."""

import os.path as osp
from unittest import mock

from absl import flags
from absl import logging
from absl.testing import absltest
from dopamine.metrics import console_collector
from dopamine.metrics import statistics_instance


class ConsoleCollectorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._tmpdir = flags.'/tmp/dopamine_tests'
    self._save_to_file = True

  def test_valid_creation(self):
    collector = console_collector.ConsoleCollector(
        self._tmpdir, save_to_file=self._save_to_file)
    self.assertEqual(collector._base_dir,
                     osp.join(self._tmpdir, 'metrics/console'))
    self.assertTrue(osp.exists(collector._base_dir))
    self.assertEqual(collector._log_file,
                     osp.join(self._tmpdir, 'metrics/console/console.log'))

  def test_valid_creation_no_base_dir(self):
    collector = console_collector.ConsoleCollector(
        None, save_to_file=self._save_to_file)
    self.assertIsNone(collector._base_dir)
    self.assertIsNone(collector._log_file)

  def test_valid_creation_no_save_to_file(self):
    collector = console_collector.ConsoleCollector(
        self._tmpdir, save_to_file=False)
    self.assertEqual(collector._base_dir,
                     osp.join(self._tmpdir, 'metrics/console'))
    self.assertTrue(osp.exists(collector._base_dir))
    self.assertIsNone(collector._log_file)

  def test_step_with_fine_grained_logging(self):
    collector = console_collector.ConsoleCollector(
        self._tmpdir, save_to_file=True)
    num_steps = 100
    logging.info = mock.MagicMock()
    collector._log_file_writer.write = mock.MagicMock()
    for i in range(num_steps):
      stat_str = (
          f'[Iteration {i}]: val = {i}\n')
      collector.write([statistics_instance.StatisticsInstance('val', i, i)])
      logging.info.assert_called_with(stat_str)
      collector._log_file_writer.write.assert_called_with(stat_str)
    self.assertEqual(logging.info.call_count, 100)
    self.assertEqual(collector._log_file_writer.write.call_count, 100)

  def test_no_write_with_unsupported_type(self):
    collector = console_collector.ConsoleCollector(
        self._tmpdir, save_to_file=True)
    num_steps = 100
    logging.info = mock.MagicMock()
    collector._log_file_writer.write = mock.MagicMock()
    for i in range(num_steps):
      collector.write([statistics_instance.StatisticsInstance(
          'val', i, i, type='unsupported')])
      logging.info.assert_not_called()
      collector._log_file_writer.write.assert_not_called()

  def test_full_run(self):
    collector = console_collector.ConsoleCollector(
        self._tmpdir, save_to_file=True)
    num_iterations = 2
    num_steps = 4
    for i in range(num_iterations):
      logging.info = mock.MagicMock()
      collector._log_file_writer.write = mock.MagicMock()
      collector._log_file_writer.close = mock.MagicMock()
      for j in range(1, num_steps * (i + 1)):
        collector.write([
            statistics_instance.StatisticsInstance('val', j, i)])
        stat_str = f'[Iteration {i}]: val = {j}\n'
        logging.info.assert_called_with(stat_str)
        collector._log_file_writer.write.assert_called_with(stat_str)
    collector.close()
    self.assertEqual(collector._log_file_writer.close.call_count, 1)


if __name__ == '__main__':
  absltest.main()

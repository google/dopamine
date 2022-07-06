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
"""Tests for dopamine.metrics.pickle_collector."""

import os.path as osp
import pickle
from unittest import mock

from absl import flags
from absl.testing import absltest
from dopamine.metrics import pickle_collector
from dopamine.metrics import statistics_instance


class PickleCollectorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._tmpdir = flags.'/tmp/dopamine_tests'

  def test_with_none_base_dir(self):
    with self.assertRaises(ValueError):
      pickle_collector.PickleCollector(None)

  def test_valid_creation(self):
    collector = pickle_collector.PickleCollector(self._tmpdir)
    self.assertEqual(collector._base_dir,
                     osp.join(self._tmpdir, 'metrics/pickle'))
    self.assertTrue(osp.exists(collector._base_dir))

  def test_write(self):
    collector = pickle_collector.PickleCollector(self._tmpdir)
    num_steps = 10
    expected_stats = {}
    pickle.dump = mock.MagicMock()
    for i in range(num_steps):
      stat = statistics_instance.StatisticsInstance('val', i, i)
      expected_stats.update({f'iteration_{i}': {'val': [i]}})
      collector.write([stat])
    self.assertEqual(pickle.dump.call_count, 0)
    self.assertEqual(expected_stats, collector._statistics)

  def test_no_write_with_unsupported_type(self):
    collector = pickle_collector.PickleCollector(self._tmpdir)
    pickle.dump = mock.MagicMock()
    for i in range(10):
      stat = statistics_instance.StatisticsInstance(
          'val', i, i, type='unsupported')
      collector.write([stat])
    self.assertEqual(pickle.dump.call_count, 0)
    self.assertEmpty(collector._statistics)

  def test_flush(self):
    collector = pickle_collector.PickleCollector(self._tmpdir)
    pickle.dump = mock.MagicMock()
    collector.write([
        statistics_instance.StatisticsInstance('val', 1, 2)
    ])
    collector.flush()
    expected_stats = {'iteration_2': {'val': [1]}}
    self.assertEqual(expected_stats, pickle.dump.call_args[0][0])
    self.assertEqual(pickle.dump.call_count, 1)
    self.assertEqual(expected_stats, collector._statistics)

  def test_full_run(self):
    collector = pickle_collector.PickleCollector(self._tmpdir)
    expected_stats = {}
    for i in range(3):
      num_steps = 3 * (i + 1)
      pickle.dump = mock.MagicMock()
      stats = []
      expected_stats[f'iteration_{i}'] = {'val_0': [], 'val_1': []}
      for j in range(1, num_steps):
        val_str = f'val_{j % 2}'
        stats.append(statistics_instance.StatisticsInstance(val_str, j, i))
        expected_stats[f'iteration_{i}'][val_str].append(j)
      collector.write(stats)
      collector.flush()
      self.assertEqual(expected_stats, pickle.dump.call_args[0][0])
      self.assertEqual(pickle.dump.call_count, 1)
    self.assertEqual(expected_stats, collector._statistics)


if __name__ == '__main__':
  absltest.main()

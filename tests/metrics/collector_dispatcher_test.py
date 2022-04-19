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
"""Tests for dopamine.metrics.collector_dispatcher."""

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from dopamine.metrics import collector
from dopamine.metrics import collector_dispatcher
from dopamine.metrics import statistics_instance


class CollectorDispatcherTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._tmpdir = flags.'/tmp/dopamine_tests'

  def test_with_no_collectors(self):
    # This test verifies that we can run successfully with no collectors.
    metrics = collector_dispatcher.CollectorDispatcher(self._tmpdir,
                                                       collectors=[])
    for i in range(4):
      stats = []
      for j in range(10):
        stats.append(statistics_instance.StatisticsInstance('val', j, i))
      metrics.write(stats)
      metrics.flush()

  def test_with_default_collectors(self):
    # This test verifies that we can run successfully with default collectors.
    metrics = collector_dispatcher.CollectorDispatcher(self._tmpdir)
    for i in range(4):
      stats = []
      for j in range(10):
        stats.append(statistics_instance.StatisticsInstance('val', j, i))
      metrics.write(stats)
      metrics.flush()

  @parameterized.named_parameters(
      dict(testcase_name='no_allowlist', allowlist=()),
      dict(testcase_name='with_allowlist', allowlist=('simple')))
  def test_with_simple_collector(self, allowlist):
    # Create a simple collector that keeps track of received statistics.
    logged_stats = []

    class SimpleCollector(collector.Collector):

      def get_name(self) -> str:
        return 'simple'

      def write(self, statistics) -> None:
        for s in statistics:
          logged_stats.append(s)

      def flush(self) -> None:
        pass

    # Create a simple collector that tracks method calls.
    counts = {
        'write': 0,
        'flush': 0,
    }

    class CountCollector(collector.Collector):

      def get_name(self) -> str:
        return 'count'

      def write(self, unused_statistics) -> None:
        counts['write'] += 1

      def flush(self) -> None:
        counts['flush'] += 1

    # Add test Collectors to the list of available collectors.
    collector_dispatcher.add_collector('simple', SimpleCollector)
    collector_dispatcher.add_collector('count', CountCollector)
    # Run a collection loop.
    metrics = collector_dispatcher.CollectorDispatcher(
        self._tmpdir, collectors=['simple', 'count'])
    expected_stats = []
    num_iterations = 4
    num_steps = 10
    for i in range(num_iterations):
      stats = []
      for j in range(num_steps):
        stats.append(statistics_instance.StatisticsInstance(
            'val', j, i))
      metrics.write(stats, collector_allowlist=allowlist)
      expected_stats += stats
    metrics.flush()
    # If using allowlist, CountCollectors write counts should not have been
    # incremewnted.
    expected_writes = 0 if allowlist else num_iterations
    self.assertEqual(
        counts,
        {'write': expected_writes, 'flush': 1})
    self.assertEqual(expected_stats, logged_stats)


if __name__ == '__main__':
  absltest.main()

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
"""Tests for dopamine.common.iteration_statistics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from dopamine.discrete_domains import iteration_statistics
import tensorflow as tf


class IterationStatisticsTest(tf.test.TestCase):

  def testMissingValue(self):
    statistics = iteration_statistics.IterationStatistics()
    with self.assertRaises(KeyError):
      _ = statistics.data_lists['missing_key']

  def testAddOneValue(self):
    statistics = iteration_statistics.IterationStatistics()

    # The statistics data structure should be empty a-priori.
    self.assertEqual(0, len(statistics.data_lists))

    statistics.append({'key1': 0})
    # We should have exactly one list, containing one value.
    self.assertEqual(1, len(statistics.data_lists))
    self.assertEqual(1, len(statistics.data_lists['key1']))
    self.assertEqual(0, statistics.data_lists['key1'][0])

  def testAddManyValues(self):
    my_pi = 3.14159

    statistics = iteration_statistics.IterationStatistics()

    # Add a number of items. Each item is added to the list corresponding to its
    # given key.
    statistics.append({'rewards': 0,
                       'nouns': 'reinforcement',
                       'angles': my_pi})
    # Add a second item to the 'nouns' list.
    statistics.append({'nouns': 'learning'})

    # There are three lists.
    self.assertEqual(3, len(statistics.data_lists))
    self.assertEqual(1, len(statistics.data_lists['rewards']))
    self.assertEqual(2, len(statistics.data_lists['nouns']))
    self.assertEqual(1, len(statistics.data_lists['angles']))

    self.assertEqual(0, statistics.data_lists['rewards'][0])
    self.assertEqual('reinforcement', statistics.data_lists['nouns'][0])
    self.assertEqual('learning', statistics.data_lists['nouns'][1])
    self.assertEqual(my_pi, statistics.data_lists['angles'][0])

if __name__ == '__main__':
  tf.compat.v1.disable_v2_behavior()
  tf.test.main()

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
"""Tests for dopamine.agents.rainbow.sum_tree."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random



from dopamine.replay_memory import sum_tree
import tensorflow as tf


class SumTreeTest(tf.test.TestCase):

  def setUp(self):
    super(SumTreeTest, self).setUp()
    self._tree = sum_tree.SumTree(capacity=100)

  def testNegativeCapacity(self):
    with self.assertRaises(ValueError,
                           msg='Sum tree capacity should be positive. Got: -1'):
      sum_tree.SumTree(capacity=-1)

  def testSetNegativeValue(self):
    with self.assertRaises(ValueError,
                           msg='Sum tree values should be nonnegative. Got -1'):
      self._tree.set(node_index=0, value=-1)

  def testSmallCapacityConstructor(self):
    tree = sum_tree.SumTree(capacity=1)
    self.assertEqual(len(tree.nodes), 1)
    tree = sum_tree.SumTree(capacity=2)
    self.assertEqual(len(tree.nodes), 2)

  def testSetValueSmallCapacity(self):
    tree = sum_tree.SumTree(capacity=1)
    tree.set(0, 1.5)
    self.assertEqual(tree.get(0), 1.5)

  def testSetValue(self):
    self._tree.set(node_index=0, value=1.0)
    self.assertEqual(self._tree.get(0), 1.0)

    # Validate that all nodes on the leftmost branch have value 1.
    for level in self._tree.nodes:
      self.assertEqual(level[0], 1.0)
      nodes_at_this_depth = len(level)
      for i in range(1, nodes_at_this_depth):
        self.assertEqual(level[i], 0.0)

  def testCapacityGreaterThanRequested(self):
    self.assertGreaterEqual(len(self._tree.nodes[-1]), 100)

  def testSampleFromEmptyTree(self):
    with self.assertRaises(Exception,
                           msg='Cannot sample from an empty sum tree.'):
      self._tree.sample()

  def testSampleWithInvalidQueryValue(self):
    self._tree.set(node_index=5, value=1.0)
    with self.assertRaises(ValueError, msg='query_value must be in [0, 1].'):
      self._tree.sample(query_value=-0.1)
    with self.assertRaises(ValueError, msg='query_value must be in [0, 1].'):
      self._tree.sample(query_value=1.1)

  def testSampleSingleton(self):
    self._tree.set(node_index=5, value=1.0)
    item = self._tree.sample()

    self.assertEqual(item, 5)

  def testSamplePairWithUnevenProbabilities(self):
    self._tree.set(node_index=2, value=1.0)
    self._tree.set(node_index=3, value=3.0)

    for _ in range(10000):
      random.seed(1)
      self.assertEqual(self._tree.sample(), 2)

  def testSamplePairWithUnevenProbabilitiesWithQueryValue(self):
    self._tree.set(node_index=2, value=1.0)
    self._tree.set(node_index=3, value=3.0)

    for _ in range(10000):
      self.assertEqual(self._tree.sample(query_value=0.1), 2)

  def testSamplingWithSeedDoesNotAffectFutureCalls(self):
    # Setting the seed here will set a deterministic random value r, which will
    # be used when sampling from the tree. Since it is scalled up by the total
    # sum value of the tree, M, we can see that r' * M + m = M, where:
    #   - M  = total sum value of the tree (total_value)
    #   - m  = value of node 3 (max_value)
    #   - r' = r + delta
    # We can then solve for M: M = m / (1 - r'), and we can set the value of the
    # node 2 to r' * M + delta, which will guarantee that
    # r * M < r' * M + delta, thereby guaranteeing that node 2 will always get
    # picked.
    seed = 1
    random.seed(seed)
    deterministic_random_value = random.random()
    max_value = 100
    delta = 0.01
    total_value = max_value / (1 - deterministic_random_value - delta)
    min_value = deterministic_random_value * total_value + delta
    self._tree.set(node_index=2, value=min_value)
    self._tree.set(node_index=3, value=max_value)
    for _ in range(10000):
      random.seed(seed)
      self.assertEqual(self._tree.sample(), 2)
    # The above loop demonstrated that there is 0 probability that node 3 gets
    # selected. The loop below demonstrates that this probability is no longer
    # 0 when the seed is not set explicitly. There is a very low probability
    # that node 2 gets selected, but to avoid flakiness, we simply assert that
    # node 3 gets selected most of the time.
    counts = {2: 0, 3: 0}
    for _ in range(10000):
      counts[self._tree.sample()] += 1
    self.assertLess(counts[2], counts[3])

  def testStratifiedSamplingFromEmptyTree(self):
    with self.assertRaises(Exception,
                           msg='Cannot sample from an empty sum tree.'):
      self._tree.stratified_sample(5)

  def testStratifiedSampling(self):
    k = 32
    for i in range(k):
      self._tree.set(node_index=i, value=1)
    samples = self._tree.stratified_sample(k)
    self.assertEqual(len(samples), k)
    for i in range(k):
      self.assertEqual(samples[i], i)

  def testMaxRecordedProbability(self):
    k = 32
    self._tree.set(node_index=0, value=0)
    self.assertEqual(self._tree.max_recorded_priority, 1)
    for i in range(1, k):
      self._tree.set(node_index=i, value=i)
      self.assertEqual(self._tree.max_recorded_priority, i)

if __name__ == '__main__':
  tf.compat.v1.disable_v2_behavior()
  tf.test.main()

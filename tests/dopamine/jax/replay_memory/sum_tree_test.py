# coding=utf-8
# Copyright 2024 The Dopamine Authors.
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
"""Tests for vectorized sum tree."""

from absl.testing import absltest
from absl.testing import parameterized
from dopamine.jax.replay_memory import sum_tree
import numpy as np


class SumTreeTest(parameterized.TestCase):

  def setUp(self):
    super(SumTreeTest, self).setUp()
    self._tree = sum_tree.SumTree(capacity=100)

  def test_negative_capacity_raises(self):
    with self.assertRaises(AssertionError):
      sum_tree.SumTree(capacity=-1)

  def test_negative_value_raises(self):
    with self.assertRaises(AssertionError):
      self._tree.set(0, -1)

  def test_set_small_capacity(self):
    tree = sum_tree.SumTree(capacity=1)
    tree.set(0, 1.5)
    self.assertEqual(tree.root, 1.5)

  def test_set_and_get_value(self):
    self._tree.set(0, 1.0)
    self.assertEqual(self._tree.get(0), 1.0)

    # Validate that all nodes on the leftmost branch have value 1.
    leaf_index = self._tree._first_leaf_offset
    while leaf_index > 0:
      leaf_index = leaf_index // 2
      self.assertEqual(self._tree._nodes[leaf_index], 1.0)

  def test_set_and_get_values_vectorized(self):
    self._tree.set(
        np.array([1, 2], dtype=np.int32),
        np.array([3.0, 4.0], dtype=np.float32),
    )
    self.assertEqual(self._tree.get(1), 3.0)
    self.assertEqual(self._tree.get(2), 4.0)
    self.assertEqual(self._tree.root, 7.0)

  def test_set_with_duplicates(self):
    self._tree.set(
        np.array([1, 1, 1, 2, 2], dtype=np.int32),
        np.array([3.0, 3.0, 3.0, 4.0, 4.0], dtype=np.float32),
    )
    self.assertEqual(self._tree.get(1), 3.0)
    self.assertEqual(self._tree.get(2), 4.0)
    self.assertEqual(self._tree.root, 7.0)

  def test_capacity_greater_than_requested(self):
    self.assertGreaterEqual(self._tree._nodes.size, 100)

  def test_query_empty_tree(self):
    with self.assertRaises(ValueError):
      self._tree.query(1.0)

  def test_query_value(self):
    self._tree.set(5, 1.0)
    self.assertEqual(self._tree.query(0.99), 5)

  def test_query_values_vectorized(self):
    #
    """.

          [2.5]
       [1.5]  [1.0]
    [0.5 1.0 0.5 0.5]
    """
    tree = sum_tree.SumTree(capacity=4)
    tree.set(
        np.array([0, 1, 2, 3], dtype=np.int32),
        np.array([0.5, 1.0, 0.5, 0.5], dtype=np.float32),
    )
    self.assertEqual(tree.root, 2.5)
    self.assertEqual(tree._depth, 3)
    self.assertEqual(tree._nodes.size, 7)
    np.testing.assert_array_equal(
        tree.query(np.array([1.5, 1.0])),
        np.array([2, 1], np.int32),
    )

  def test_update_sum_values(self):
    #
    """.

          [2.5]
       [1.5]  [1.0]
    [0.5 1.0 0.5 0.5]
    """
    tree = sum_tree.SumTree(capacity=4)
    tree.set(
        np.array([0, 1, 2, 3], dtype=np.int32),
        np.array([0.5, 1.0, 0.5, 0.5], dtype=np.float32),
    )
    tree.set(0, 0.25)
    self.assertEqual(tree.root, 2.25)
    self.assertEqual(tree.query(0.249), 0)
    self.assertEqual(tree.query(0.5), 1)
    self.assertEqual(tree.query(1.25), 2)

  def test_query_values_vectorized_large_tree(self):
    #
    """.

              [8]
         [4]         [4]
      [2]   [2]   [2]   [2]
    [1, 1, 1, 1, 1, 1, 1, 1]
    """
    tree = sum_tree.SumTree(capacity=8)
    tree.set(
        np.arange(8, dtype=np.int32),
        np.ones((8,), dtype=np.float32),
    )
    self.assertEqual(tree.root, 8.0)
    self.assertEqual(tree._depth, 4)
    self.assertEqual(tree._nodes.size, 15)
    np.testing.assert_array_equal(
        tree.query(np.arange(8, dtype=np.int32)),
        np.arange(8, dtype=np.int32),
    )

  def test_serialization(self):
    self._tree.set(5, 1.0)
    state_dict = self._tree.to_state_dict()
    np.testing.assert_array_equal(state_dict['nodes'], self._tree._nodes)
    self._tree.from_state_dict(state_dict)
    np.testing.assert_array_equal(state_dict['nodes'], self._tree._nodes)
    self.assertEqual(self._tree.root, 1.0)
    self.assertEqual(self._tree.get(5), 1.0)

  def test_clear(self):
    self._tree.set(5, 1.0)
    self._tree.clear()
    self.assertEqual(self._tree.root, 0.0)
    self.assertEqual(self._tree.get(5), 0.0)

  def test_max_recorded_priority(self):
    k = 32
    self._tree.set(0, 0)
    self.assertEqual(self._tree.max_recorded_priority, 1)
    for i in range(1, k):
      self._tree.set(i, i)
      self.assertEqual(self._tree.max_recorded_priority, i)


if __name__ == '__main__':
  absltest.main()

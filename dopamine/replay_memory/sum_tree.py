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
"""A sum tree data structure.

Used for prioritized experience replay. See prioritized_replay_buffer.py
and Schaul et al. (2015).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random

import numpy as np


class SumTree(object):
  """A sum tree data structure for storing replay priorities.

  A sum tree is a complete binary tree whose leaves contain values called
  priorities. Internal nodes maintain the sum of the priorities of all leaf
  nodes in their subtree.

  For capacity = 4, the tree may look like this:

               +---+
               |2.5|
               +-+-+
                 |
         +-------+--------+
         |                |
       +-+-+            +-+-+
       |1.5|            |1.0|
       +-+-+            +-+-+
         |                |
    +----+----+      +----+----+
    |         |      |         |
  +-+-+     +-+-+  +-+-+     +-+-+
  |0.5|     |1.0|  |0.5|     |0.5|
  +---+     +---+  +---+     +---+

  This is stored in a list of numpy arrays:
  self.nodes = [ [2.5], [1.5, 1], [0.5, 1, 0.5, 0.5] ]

  For conciseness, we allocate arrays as powers of two, and pad the excess
  elements with zero values.

  This is similar to the usual array-based representation of a complete binary
  tree, but is a little more user-friendly.
  """

  def __init__(self, capacity):
    """Creates the sum tree data structure for the given replay capacity.

    Args:
      capacity: int, the maximum number of elements that can be stored in this
        data structure.

    Raises:
      ValueError: If requested capacity is not positive.
    """
    assert isinstance(capacity, int)
    if capacity <= 0:
      raise ValueError('Sum tree capacity should be positive. Got: {}'.
                       format(capacity))

    self.nodes = []
    tree_depth = int(math.ceil(np.log2(capacity)))
    level_size = 1
    for _ in range(tree_depth + 1):
      nodes_at_this_depth = np.zeros(level_size)
      self.nodes.append(nodes_at_this_depth)

      level_size *= 2

    self.max_recorded_priority = 1.0

  def _total_priority(self):
    """Returns the sum of all priorities stored in this sum tree.

    Returns:
      float, sum of priorities stored in this sum tree.
    """
    return self.nodes[0][0]

  def sample(self, query_value=None):
    """Samples an element from the sum tree.

    Each element has probability p_i / sum_j p_j of being picked, where p_i is
    the (positive) value associated with node i (possibly unnormalized).

    Args:
      query_value: float in [0, 1], used as the random value to select a
      sample. If None, will select one randomly in [0, 1).

    Returns:
      int, a random element from the sum tree.

    Raises:
      Exception: If the sum tree is empty (i.e. its node values sum to 0), or if
        the supplied query_value is larger than the total sum.
    """
    if self._total_priority() == 0.0:
      raise Exception('Cannot sample from an empty sum tree.')

    if query_value and (query_value < 0. or query_value > 1.):
      raise ValueError('query_value must be in [0, 1].')

    # Sample a value in range [0, R), where R is the value stored at the root.
    query_value = random.random() if query_value is None else query_value
    query_value *= self._total_priority()

    # Now traverse the sum tree.
    node_index = 0
    for nodes_at_this_depth in self.nodes[1:]:
      # Compute children of previous depth's node.
      left_child = node_index * 2

      left_sum = nodes_at_this_depth[left_child]
      # Each subtree describes a range [0, a), where a is its value.
      if query_value < left_sum:  # Recurse into left subtree.
        node_index = left_child
      else:  # Recurse into right subtree.
        node_index = left_child + 1
        # Adjust query to be relative to right subtree.
        query_value -= left_sum

    return node_index

  def stratified_sample(self, batch_size):
    """Performs stratified sampling using the sum tree.

    Let R be the value at the root (total value of sum tree). This method will
    divide [0, R) into batch_size segments, pick a random number from each of
    those segments, and use that random number to sample from the sum_tree. This
    is as specified in Schaul et al. (2015).

    Args:
      batch_size: int, the number of strata to use.
    Returns:
      list of batch_size elements sampled from the sum tree.

    Raises:
      Exception: If the sum tree is empty (i.e. its node values sum to 0).
    """
    if self._total_priority() == 0.0:
      raise Exception('Cannot sample from an empty sum tree.')

    bounds = np.linspace(0., 1., batch_size + 1)
    assert len(bounds) == batch_size + 1
    segments = [(bounds[i], bounds[i+1]) for i in range(batch_size)]
    query_values = [random.uniform(x[0], x[1]) for x in segments]
    return [self.sample(query_value=x) for x in query_values]

  def get(self, node_index):
    """Returns the value of the leaf node corresponding to the index.

    Args:
      node_index: The index of the leaf node.
    Returns:
      The value of the leaf node.
    """
    return self.nodes[-1][node_index]

  def set(self, node_index, value):
    """Sets the value of a leaf node and updates internal nodes accordingly.

    This operation takes O(log(capacity)).
    Args:
      node_index: int, the index of the leaf node to be updated.
      value: float, the value which we assign to the node. This value must be
        nonnegative. Setting value = 0 will cause the element to never be
        sampled.

    Raises:
      ValueError: If the given value is negative.
    """
    if value < 0.0:
      raise ValueError('Sum tree values should be nonnegative. Got {}'.
                       format(value))
    self.max_recorded_priority = max(value, self.max_recorded_priority)

    delta_value = value - self.nodes[-1][node_index]

    # Now traverse back the tree, adjusting all sums along the way.
    for nodes_at_this_depth in reversed(self.nodes):
      # Note: Adding a delta leads to some tolerable numerical inaccuracies.
      nodes_at_this_depth[node_index] += delta_value
      node_index //= 2

    assert node_index == 0, ('Sum tree traversal failed, final node index '
                             'is not 0.')

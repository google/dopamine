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
"""A class for storing iteration-specific metrics.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class IterationStatistics(object):
  """A class for storing iteration-specific metrics.

  The internal format is as follows: we maintain a mapping from keys to lists.
  Each list contains all the values corresponding to the given key.

  For example, self.data_lists['train_episode_returns'] might contain the
    per-episode returns achieved during this iteration.

  Attributes:
    data_lists: dict mapping each metric_name (str) to a list of said metric
      across episodes.
  """

  def __init__(self):
    self.data_lists = {}

  def append(self, data_pairs):
    """Add the given values to their corresponding key-indexed lists.

    Args:
      data_pairs: A dictionary of key-value pairs to be recorded.
    """
    for key, value in data_pairs.items():
      if key not in self.data_lists:
        self.data_lists[key] = []
      self.data_lists[key].append(value)

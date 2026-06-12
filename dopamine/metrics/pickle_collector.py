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
"""Collector class for saving iteration statistics to a msgpack file.

SECURITY FIX: Replaced pickle serialization with msgpack to eliminate
Remote Code Execution (RCE) vulnerability. The class name is retained
for backward compatibility but the on-disk format is now msgpack.
"""

import collections
import functools
import os.path as osp
from typing import Sequence

from dopamine.metrics import collector
from dopamine.metrics import statistics_instance
import msgpack
import msgpack_numpy
import tensorflow as tf


def _pack(data):
  """Serialize data to msgpack bytes with numpy support."""
  return msgpack.packb(data, default=msgpack_numpy.encode, use_bin_type=True)


class PickleCollector(collector.Collector):
  """Collector class for saving iteration statistics to a msgpack file.

  The class is named PickleCollector for backward compatibility, but it now
  writes msgpack files instead of pickle files to prevent RCE attacks.
  """

  def __init__(self, base_dir: str):
    if base_dir is None:
      raise ValueError('Must specify a base directory for PickleCollector.')
    super().__init__(base_dir)
    listdict = functools.partial(collections.defaultdict, list)
    self._statistics = collections.defaultdict(listdict)
    self._file_number = 0

  def get_name(self) -> str:
    return 'pickle'

  def write(
      self, statistics: Sequence[statistics_instance.StatisticsInstance]
  ) -> None:
    """Accumulates statistics for the current iteration.

    Args:
      statistics: Sequence of StatisticsInstance objects to record.
    """
    for s in statistics:
      if not self.check_type(s.type):
        continue
      self._statistics[f'iteration_{s.step}'][s.name].append(s.value)

  def flush(self):
    """Writes accumulated statistics to a msgpack file and resets state."""
    msgpack_file = osp.join(
        self._base_dir, f'pickle_{self._file_number}.msgpack'
    )
    with tf.io.gfile.GFile(msgpack_file, 'wb') as f:
      f.write(_pack(dict(self._statistics)))
    self._file_number += 1
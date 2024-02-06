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
"""Collector class for saving iteration statistics to a pickle file."""

import collections
import functools
import os.path as osp
import pickle
from typing import Sequence

from dopamine.metrics import collector
from dopamine.metrics import statistics_instance
import tensorflow as tf


class PickleCollector(collector.Collector):
  """Collector class for reporting statistics to the console."""

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
    # This Collector is trying to write metrics as close as possible to what
    # is currently written by the Dopamine Logger, so as to be as compatible
    # with user's plotting setups.
    for s in statistics:
      if not self.check_type(s.type):
        continue
      self._statistics[f'iteration_{s.step}'][s.name].append(s.value)

  def flush(self):
    pickle_file = osp.join(self._base_dir, f'pickle_{self._file_number}.pkl')
    with tf.io.gfile.GFile(pickle_file, 'w') as f:
      pickle.dump(self._statistics, f, protocol=pickle.HIGHEST_PROTOCOL)
    self._file_number += 1

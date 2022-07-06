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
"""Collector class for exporting statistics to Tensorboard."""

from typing import Sequence
from dopamine.metrics import collector
from dopamine.metrics import statistics_instance
import tensorflow as tf


class TensorboardCollector(collector.Collector):
  """Collector class for reporting statistics on Tensorboard."""

  def __init__(self, base_dir: str):
    if not isinstance(base_dir, str):
      raise ValueError(
          'Must specify a base directory for TensorboardCollector.')
    super().__init__(base_dir)
    self.summary_writer = tf.summary.create_file_writer(self._base_dir)

  def get_name(self) -> str:
    return 'tensorboard'

  def write(
      self,
      statistics: Sequence[statistics_instance.StatisticsInstance]) -> None:
    with self.summary_writer.as_default():
      for s in statistics:
        if not self.check_type(s.type):
          continue
        tf.summary.scalar(s.name, s.value, step=s.step)

  def flush(self):
    self.summary_writer.flush()

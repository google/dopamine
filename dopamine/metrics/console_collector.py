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
"""Collector class for reporting statistics to the console."""

import os.path as osp
from typing import Sequence, Union

from absl import logging
from dopamine.metrics import collector
from dopamine.metrics import statistics_instance
import gin
import tensorflow as tf


@gin.configurable(allowlist=['save_to_file'])
class ConsoleCollector(collector.Collector):
  """Collector class for reporting statistics to the console."""

  def __init__(self, base_dir: Union[str, None], save_to_file: bool = True):
    super().__init__(base_dir)
    if self._base_dir is not None and save_to_file:
      self._log_file = osp.join(self._base_dir, 'console.log')
      self._log_file_writer = tf.io.gfile.GFile(self._log_file, 'w')
    else:
      self._log_file = None

  def get_name(self) -> str:
    return 'console'

  def write(
      self, statistics: Sequence[statistics_instance.StatisticsInstance]
  ) -> None:
    step_string = ''
    for s in statistics:
      if not self.check_type(s.type):
        continue
      step_string += f'[Iteration {s.step}]: {s.name} = {s.value}\n'
    # Only write out if step_string is non-empty
    if step_string:
      logging.info(step_string)
      if self._log_file is not None:
        self._log_file_writer.write(step_string)

  def close(self) -> None:
    if self._log_file is not None:
      self._log_file_writer.close()  # pytype: disable=attribute-error  # trace-all-classes

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
"""Base class for metric collectors.

Each Collector should subclass this base class, as the CollectorDispatcher
object expects objects of type Collector.

The methods to implement are:
  - `get_name`: a unique identifier for subdirectory creation.
  - `pre_training`: called once before training begins.
  - `step`: called once for each training step. The parameter is an object of
    type `StatisticsInstance` which contains the statistics of the current
    training step.
  - `end_training`: called once at the end of training, and passes in a
    `StatisticsInstance` containing the statistics of the latest training step.
"""

import abc
import os.path as osp
from typing import Optional, Sequence

from dopamine.metrics import statistics_instance
import tensorflow as tf


class Collector(abc.ABC):
  """Abstract class for defining metric collectors."""

  def __init__(
      self, base_dir: Optional[str], extra_supported_types: Sequence[str] = ()
  ):
    if base_dir is not None:
      self._base_dir = osp.join(base_dir, 'metrics', self.get_name())
      # Try to create logging directory.
      try:
        tf.io.gfile.makedirs(self._base_dir)
      except tf.errors.PermissionDeniedError:
        # If it already exists, ignore exception.
        pass
    else:
      self._base_dir = None
    self._supported_types = ['scalar'] + list(extra_supported_types)

  @abc.abstractmethod
  def get_name(self) -> str:
    pass

  def check_type(self, data_type: str) -> bool:
    return data_type in self._supported_types

  @abc.abstractmethod
  def write(
      self, statistics: Sequence[statistics_instance.StatisticsInstance]
  ) -> None:
    pass

  def flush(self) -> None:
    pass

  def close(self) -> None:
    pass

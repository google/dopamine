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
"""Class that runs a list of Collectors for metrics reporting.

This class is what should be called from the main binary and will call each of
the specified collectors for metrics reporting.

Each metric collector can be further configured via gin bindings. The
constructor for each desired collector should be passed in as a list when
creating this object. All of the collectors are expected to be subclasses of the
`Collector` base class (defined in `collector.py`).

Example configuration:
```
metrics = CollectorDispatcher(base_dir, num_actions, list_of_constructors)
metrics.pre_training()
for i in range(training_steps):
  ...
  metrics.step(statistics)
metrics.end_training(statistics)
```

The statistics are passed in as a dict that contains
and contains the raw performance statistics for the current iteration. All
processing (such as averaging) will be handled by each of the individual
collectors.
"""

from typing import Callable, Optional, Sequence

from absl import logging
from dopamine.metrics import collector
from dopamine.metrics import console_collector
from dopamine.metrics import pickle_collector
from dopamine.metrics import statistics_instance
from dopamine.metrics import tensorboard_collector
import gin

AVAILABLE_COLLECTORS = {
    'console': console_collector.ConsoleCollector,
    'pickle': pickle_collector.PickleCollector,
    'tensorboard': tensorboard_collector.TensorboardCollector,
}


CollectorConstructorType = Callable[[str], collector.Collector]


def add_collector(name: str, constructor: CollectorConstructorType) -> None:
  AVAILABLE_COLLECTORS.update({name: constructor})


@gin.configurable
class CollectorDispatcher(object):
  """Class for collecting and reporting Dopamine metrics."""

  def __init__(
      self,
      base_dir: Optional[str],
      # TODO(psc): Consider using sets instead.
      collectors: Sequence[str] = ('console', 'pickle', 'tensorboard'),
  ):
    self._collectors = []
    for c in collectors:
      if c not in AVAILABLE_COLLECTORS:
        logging.warning('Collector %s not recognized, ignoring.', c)
        continue
      self._collectors.append(AVAILABLE_COLLECTORS[c](base_dir))
      logging.info('Added collector %s.', c)

  def write(
      self,
      statistics: Sequence[statistics_instance.StatisticsInstance],
      collector_allowlist: Sequence[str] = (),
  ) -> None:
    """Write a list of statistics to various collectors.

    Args:
      statistics: A list of of StatisticsInstances to write.
      collector_allowlist: A list of Collectors to include in this call to
        write. This is to enable users to, for instance, which Collectors will
        be used to write fine-grained statistics. If collector_allowlist is
        empty, all available Collectors will be called.
    """
    for c in self._collectors:
      if collector_allowlist and c.get_name() not in collector_allowlist:
        continue
      c.write(statistics)

  def flush(self) -> None:
    for c in self._collectors:
      c.flush()

  def close(self) -> None:
    for c in self._collectors:
      c.close()

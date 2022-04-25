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
"""Tests for dopamine.metrics.collector."""

import os.path as osp

from absl import flags
from absl.testing import absltest
from dopamine.metrics import collector


# A simple subclass that implements the abstract methods.
class SimpleCollector(collector.Collector):

  def get_name(self) -> str:
    return 'simple'

  def write(self, unused_statistics) -> None:
    pass

  def flush(self, unused_statistics) -> None:
    pass

  def close(self) -> None:
    pass


class CollectorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._tmpdir = flags.'/tmp/dopamine_tests'

  def test_instantiate_abstract_class(self):
    # It is not possible to instantiate Collector as it has abstract methods.
    with self.assertRaises(TypeError):
      collector.Collector(self._tmpdir)

  def test_valid_subclass(self):
    simple_collector = SimpleCollector(self._tmpdir)
    self.assertEqual(simple_collector._base_dir,
                     osp.join(self._tmpdir, 'metrics/simple'))
    self.assertTrue(osp.exists(simple_collector._base_dir))

  def test_valid_subclass_with_no_basedir(self):
    simple_collector = SimpleCollector(None)
    self.assertIsNone(simple_collector._base_dir)


if __name__ == '__main__':
  absltest.main()

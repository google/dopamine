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
"""Tests for threading_utils.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mock
import tempfile
import threading

from dopamine.agents.dqn import dqn_agent
from dopamine.utils import threading_utils
import tensorflow as tf
from tensorflow import test


def _get_internal_name(attr):
  thread_id = str(threading.current_thread().ident)
  return '__' + attr + '_' + thread_id


def _create_mock_object(**kwargs):
  @threading_utils.local_attributes(kwargs.keys())
  class _MockClass(object):

    def __init__(self):
      threading_utils.initialize_local_attributes(self, **kwargs)
  return _MockClass()


class ThreadsTest(test.TestCase):
  """Unit tests for threading utils."""

  def test_default_value_is_added(self):
    """Tests that the default value is properly set by the helper."""
    obj = mock.Mock()
    threading_utils.initialize_local_attributes(obj, attr=3)
    self.assertEqual(obj.attr_default, 3)

  def test_multiple_default_values_are_set(self):
    """Tests that multiple default values are properly set by the helper."""
    obj = mock.Mock()
    threading_utils.initialize_local_attributes(obj, attr1=3, attr2=4)
    self.assertEqual(obj.attr1_default, 3)
    self.assertEqual(obj.attr2_default, 4)

  def test_attribute_default_value_is_called(self):
    """Tests that getter properly uses the default value."""
    MockClass = threading_utils.local_attributes(['attr'])(
        type('MockClass', (object,), {}))
    obj = MockClass()
    obj.attr_default = 'default-value'
    self.assertEqual(obj.attr, 'default-value')

  def test_default_value_is_read(self):
    """Tests that getter properly initializes the local value."""
    MockClass = threading_utils.local_attributes(['attr'])(
        type('MockClass', (object,), {}))
    obj = MockClass()
    obj.attr_default = 'default-value'
    obj.attr  # pylint: disable=pointless-statement
    self.assertEqual(getattr(obj, _get_internal_name('attr')), 'default-value')

  def test_internal_attribute_is_read(self):
    """Tests that getter properly uses the internal value."""
    MockClass = threading_utils.local_attributes(['attr'])(
        type('MockClass', (object,), {}))
    obj = MockClass()
    setattr(obj, _get_internal_name('attr'), 'intenal-value')
    self.assertEqual(obj.attr, 'intenal-value')

  def test_internal_attribute_is_set(self):
    """Tests that setter properly sets the internal value."""
    MockClass = threading_utils.local_attributes(['attr'])(
        type('MockClass', (object,), {}))
    obj = MockClass()
    obj.attr = 'internal-value'
    self.assertEqual(getattr(obj, _get_internal_name('attr')), 'internal-value')

  def test_internal_value_over_default(self):
    """Tests that getter uese internal value over default one."""
    MockClass = threading_utils.local_attributes(['attr'])(
        type('MockClass', (object,), {}))
    obj = MockClass()
    obj.attr_default = 'default-value'
    setattr(obj, _get_internal_name('attr'), 'internal-value')
    self.assertEqual(obj.attr, 'internal-value')

  def test_multiple_attributes(self):
    """Tests the class decorator with multiple local attributes."""
    MockClass = threading_utils.local_attributes(['attr1', 'attr2'])(
        type('MockClass', (object,), {}))
    obj = MockClass()
    obj.attr1 = 10
    obj.attr2 = 20
    setattr(obj, _get_internal_name('attr1'), 1)
    setattr(obj, _get_internal_name('attr2'), 2)

  def test_callable_attribute(self):
    """Tests that internal value is properly called with callable attribute."""
    MockClass = threading_utils.local_attributes(['attr'])(
        type('MockClass', (object,), {}))
    obj = MockClass()
    internal_attr = mock.Mock()
    setattr(obj, _get_internal_name('attr'), internal_attr)
    obj.attr.callable_method()
    internal_attr.callable_method.assert_called_once()


class DQNIntegrationTest(test.TestCase):
  """Integration test for DQNAgent and threading utils."""

  def test_bundling(self):
    """Tests that local values are poperly updated when reading a checkpoint."""
    with tf.Session() as sess:
      agent = agent = dqn_agent.DQNAgent(sess, 3, observation_shape=(2, 2))
      sess.run(tf.global_variables_initializer())
      agent.state = 'state_val'
      self.assertEqual(
          getattr(agent, _get_internal_name('state')), 'state_val')
      test_dir = tempfile.mkdtemp()
      bundle = agent.bundle_and_checkpoint(test_dir, iteration_number=10)
      self.assertIn('state', bundle)
      self.assertEqual(bundle['state'], 'state_val')
      bundle['state'] = 'new_state_val'

      agent.unbundle(test_dir, iteration_number=10, bundle_dictionary=bundle)
      self.assertEqual(agent.state, 'new_state_val')
      self.assertEqual(
          getattr(agent, _get_internal_name('state')), 'new_state_val')


if __name__ == '__main__':
  test.main()

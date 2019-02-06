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

import tempfile

import numpy as np
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

  def test_main_thread(self):
    obj = _create_mock_object(attr=3)
    self.assertEqual(obj.attr_default, 3)
    self.assertEqual(obj.attr, 3)
    obj.attr = 5
    internal_name = _get_internal_name('attr')
    self.assertTrue(hasattr(obj, internal_name))
    self.assertEqual(getattr(obj, internal_name), 5)

  def test_multiple_attributes(self):
    obj = _create_mock_object(attr_1=1, attr_2=2)
    self.assertEqual(obj.attr_1, 1)
    self.assertEqual(getattr(obj, _get_internal_name('attr_1')), 1)
    self.assertEqual(obj.attr_2, 2)
    self.assertEqual(getattr(obj, _get_internal_name('attr_2')), 2)

  def test_np_array(self):
    obj = _create_mock_object(attr=np.zeros((2, 3)))
    self.assertTrue(obj.attr is not None)
    internal_name = _get_internal_name('attr')
    obj.attr.fill(1)
    self.assertEqual(obj.attr.min(), 1)


class DQNTest(test.TestCase):

  def test_dqn_agent(self):
    agent = dqn_agent.DQNAgent(self.cached_session(), 3)
    self.assertTrue(agent.state is not None)
    self.assertEqual(agent.state.shape, (1, 84, 84, 4))

  def test_bundling(self):
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

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
"""Common testing utilities shared across agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import mock
import tensorflow as tf


class MockReplayBuffer(object):
  """Mock ReplayBuffer to verify the way the agent interacts with it."""

  def __init__(self):
    with tf.variable_scope('MockReplayBuffer', reuse=tf.AUTO_REUSE):
      self.add = mock.Mock()
      self.memory = mock.Mock()
      self.memory.add_count = 0

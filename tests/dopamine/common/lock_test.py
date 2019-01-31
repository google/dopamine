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
"""Tests for lock_decorator.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dopamine.common import lock
from tensorflow import test


class _MockLock(object):
  """Mock lock for testing purposes."""

  def __enter__(self, *args, **kwargs):
    """Locks the lock."""
    raise ValueError('Lock is locked.')

  def __exit__(slef, *args, **kwargs):
    pass


class _MockClass(object):
  """Mock class to test the lock againts."""

  @lock.locked_method
  def mock_method(self):
    pass


class LockDecoratorTest(test.TestCase):
  """Runs tests for lock_decorator function."""

  def test_locks_applies(self):
    """Tests that the lock properly applies to a given function."""

    mock_object = _MockClass()
    mock_object._lock = _MockLock()
    with self.assertRaisesRegexp(ValueError, 'Lock is locked.'):
      mock_object.mock_method()

  def test_no_lock_attribute(self):
    mock_object = _MockClass()
    with self.assertRaisesRegexp(
        AttributeError, r'Object .* expected to have a `_lock` attribute.'):
      mock_object.mock_method()


if __name__ == '__main__':
  test.main()

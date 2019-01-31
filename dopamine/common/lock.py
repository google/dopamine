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
"""Creates a function decorator to protect execution with a lock."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

_LOCK_ATTR_NAME = '_lock'


class _MockLock(object):
  """Mock lock for testing purposes."""

  def __init__(self):
    self._blocked = None

  def __enter__(self, *args, **kwargs):
    """Locks the lock."""
    if self._blocked:
      raise ValueError('Lock is locked.')
    self._blocked = True

  def __exit__(self, *args, **kwargs):
    """Purposely leaves the lock locked to be `manually` unlocked."""
    self._blocked = False


class LockedClass(object):
  def __init__(self, *args, **kwargs):
    if hasattr(self, _LOCK_ATTR_NAME):
      raise ValueError('Object already has a `{}` attribute.')
    setattr(self, _LOCK_ATTR_NAME, _MockLock())


def lock_decorator(fn):
  """Wraps a class's method so it's locked.

  Args:
    fn: Object's method with the following signature:
      * Args:
        * self: Instance of the class.
        * *args: Additional positional arguments.
        * **kwargs: Additional keyword arguments.
      Note that the instance must have a `_lock` attribute.

  Returns:
    A function with same signature as the input function.
  """
  def _decorated(self, *args, **kwargs):
    lock = getattr(self, _LOCK_ATTR_NAME, None)
    if not lock:
      raise AttributeError(
          'Object {} expected to have a `{}` attribute.'.format(
              self, _LOCK_ATTR_NAME))
    with lock:  # pylint: disable=protected-access
      return fn(self, *args, **kwargs)
  return _decorated

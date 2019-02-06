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
"""Decorator to decouple object attributes and make them local to threads."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading


def _get_internal_name(name):
  """Returns the thread local name of an attribute."""
  return '__' + name + '_' + str(threading.current_thread().ident)


def _get_default_value_name(name):
  """Returns the global default value of an attribute."""
  return name + '_default'


def _add_property(cls, attr_name):
  """Adds a property to a given class.

  The a setter, getter and deleter are added to the given class with to the
  provided attribute name.
  These methods actually apply to an intenal variable that is thread-specific.
  Hence the result of these methods depends on the local thread.

  Note that when the getter is called and the local attribute is not found the
  getter will initialize the local value to the global default value.
  See `initialize_local_attributes` for more details.

  Args:
    cls: A class to add the poperty to.
    attr_name: str, name of the property to create.
  """
  def _set(self, val):
    setattr(self, _get_internal_name(attr_name), val)

  def _get(self):
    if not hasattr(self, _get_internal_name(attr_name)):
      _set(self, getattr(self, _get_default_value_name(attr_name)))
    return getattr(self, _get_internal_name(attr_name))

  def _del(self):
    delattr(self, _get_internal_name(attr_name))
  setattr(cls, attr_name, property(_get, _set, _del))


def local_attributes(attributes):
  """Creates a decorator that add properties to the decorated class.

  Args:
    attributes: List[str], names of the wrapped attributes to add to the class.

  Returns:
    A class decorator.
  """
  def _decorator(cls):
    for attr_name in attributes:
      _add_property(cls, attr_name)
    return cls
  return _decorator


def initialize_local_attributes(obj, **kwargs):
  """Sets global default values for local attributes.

  Each attribute has a global default value and local values that are specific
  to each thread.
  In each thread, the first time the getter is called it is initialized to the
  global default value. This helper function is to set these default value.

  Example of usage:
    ```python
    @local_attributes(['attr'])
    class MyClass(object):

      def __init__(self, attr_default_value):
        initialize_local_attributes(self, attr=attr_default_value)
    ```
  Args:
    obj: The object that has the local attributes.
    **kwargs: The default value for each local attributes.
  """
  for key, val in kwargs.items():
    setattr(obj, _get_default_value_name(key), val)

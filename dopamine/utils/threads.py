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
"""Decorator to decouple attributes per threads."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading


def _get_internal_name(name):
  return '__' + name + '_' + str(threading.current_thread().ident)


def _get_default_value_name(name):
  return name + '_default'


def _add_property(cls, attr_name):
  def _set(self, val):
    setattr(self, _get_internal_name(attr_name), val)

  def _get(self):
    if not hasattr(self, _get_internal_name(attr_name)):
      _set(self, getattr(self, _get_default_value_name(attr_name)))
    return getattr(self, _get_internal_name(attr_name))

  def _del(self):
    delattr(self, _get_internal_name(attr_name))
  setattr(cls, attr_name, property(_get, _set, _del))


class local_attributes(object):

  def __init__(self, attributes):
    self._attributes = attributes

  def __call__(self, cls):
    for attr_name in self._attributes:
      _add_property(cls, attr_name)
    return cls


class LocalAttributes(object):

  def __init__(self, **kwargs):
    for key, val in kwargs.items():
      setattr(self, _get_default_value_name(key), val)

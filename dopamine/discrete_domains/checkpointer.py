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
# A lightweight logging mechanism for dopamine agents.
# SECURITY FIX: Replaced pickle serialization with msgpack to eliminate
# Remote Code Execution (RCE) vulnerability.

import os

from absl import logging
import gin
import msgpack
import msgpack_numpy
import tensorflow as tf


def _pack(data):
  return msgpack.packb(data, default=msgpack_numpy.encode, use_bin_type=True)


@gin.configurable
class Logger(object):
  """Class for maintaining a dictionary of data to log."""

  def __init__(self, logging_dir, logs_duration=4):
    self.data = {}
    self._logging_enabled = True
    self._logs_duration = logs_duration

    if not logging_dir:
      logging.info('Logging directory not specified, will not log.')
      self._logging_enabled = False
      return

    try:
      tf.io.gfile.makedirs(logging_dir)
    except tf.errors.PermissionDeniedError:
      pass

    if not tf.io.gfile.exists(logging_dir):
      logging.warning(
          'Could not create directory %s, logging will be disabled.',
          logging_dir,
      )
      self._logging_enabled = False
      return

    self._logging_dir = logging_dir

  def __setitem__(self, key, value):
    if self._logging_enabled:
      self.data[key] = value

  def _generate_filename(self, filename_prefix, iteration_number):
    filename = '{}_{}'.format(filename_prefix, iteration_number)
    return os.path.join(self._logging_dir, filename)

  def log_to_file(self, filename_prefix, iteration_number):
    if not self._logging_enabled:
      logging.warning('Logging is disabled.')
      return

    log_file = self._generate_filename(filename_prefix, iteration_number)
    with tf.io.gfile.GFile(log_file, 'wb') as fout:
      fout.write(_pack(self.data))

    stale_iteration_number = iteration_number - self._logs_duration
    if stale_iteration_number >= 0:
      stale_file = self._generate_filename(
          filename_prefix, stale_iteration_number
      )
      try:
        tf.io.gfile.remove(stale_file)
      except tf.errors.NotFoundError:
        pass

  def is_logging_enabled(self):
    return self._logging_enabled
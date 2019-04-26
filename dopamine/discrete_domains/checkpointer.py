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
"""A checkpointing mechanism for Dopamine agents.

This Checkpointer expects a base directory where checkpoints for different
iterations are stored. Specifically, Checkpointer.save_checkpoint() takes in
as input a dictionary 'data' to be pickled to disk. At each iteration, we
write a file called 'cpkt.#', where # is the iteration number. The
Checkpointer also cleans up old files, maintaining up to the CHECKPOINT_DURATION
most recent iterations.

The Checkpointer writes a sentinel file to indicate that checkpointing was
globally successful. This means that all other checkpointing activities
(saving the Tensorflow graph, the replay buffer) should be performed *prior*
to calling Checkpointer.save_checkpoint(). This allows the Checkpointer to
detect incomplete checkpoints.

#### Example

After running 10 iterations (numbered 0...9) with base_directory='/checkpoint',
the following files will exist:
```
  /checkpoint/cpkt.6
  /checkpoint/cpkt.7
  /checkpoint/cpkt.8
  /checkpoint/cpkt.9
  /checkpoint/sentinel_checkpoint_complete.6
  /checkpoint/sentinel_checkpoint_complete.7
  /checkpoint/sentinel_checkpoint_complete.8
  /checkpoint/sentinel_checkpoint_complete.9
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import tensorflow as tf

CHECKPOINT_DURATION = 4


def get_latest_checkpoint_number(base_directory):
  """Returns the version number of the latest completed checkpoint.

  Args:
    base_directory: str, directory in which to look for checkpoint files.

  Returns:
    int, the iteration number of the latest checkpoint, or -1 if none was found.
  """
  glob = os.path.join(base_directory, 'sentinel_checkpoint_complete.*')
  def extract_iteration(x):
    return int(x[x.rfind('.') + 1:])
  try:
    checkpoint_files = tf.gfile.Glob(glob)
  except tf.errors.NotFoundError:
    return -1
  try:
    latest_iteration = max(extract_iteration(x) for x in checkpoint_files)
    return latest_iteration
  except ValueError:
    return -1


class Checkpointer(object):
  """Class for managing checkpoints for Dopamine agents.
  """

  def __init__(self, base_directory, checkpoint_file_prefix='ckpt',
               checkpoint_frequency=1):
    """Initializes Checkpointer.

    Args:
      base_directory: str, directory where all checkpoints are saved/loaded.
      checkpoint_file_prefix: str, prefix to use for naming checkpoint files.
      checkpoint_frequency: int, the frequency at which to checkpoint.

    Raises:
      ValueError: if base_directory is empty, or not creatable.
    """
    if not base_directory:
      raise ValueError('No path provided to Checkpointer.')
    self._checkpoint_file_prefix = checkpoint_file_prefix
    self._checkpoint_frequency = checkpoint_frequency
    self._base_directory = base_directory
    try:
      tf.gfile.MakeDirs(base_directory)
    except tf.errors.PermissionDeniedError:
      # We catch the PermissionDeniedError and issue a more useful exception.
      raise ValueError('Unable to create checkpoint path: {}.'.format(
          base_directory))

  def _generate_filename(self, file_prefix, iteration_number):
    """Returns a checkpoint filename from prefix and iteration number."""
    filename = '{}.{}'.format(file_prefix, iteration_number)
    return os.path.join(self._base_directory, filename)

  def _save_data_to_file(self, data, filename):
    """Saves the given 'data' object to a file."""
    with tf.gfile.GFile(filename, 'w') as fout:
      pickle.dump(data, fout)

  def save_checkpoint(self, iteration_number, data):
    """Saves a new checkpoint at the current iteration_number.

    Args:
      iteration_number: int, the current iteration number for this checkpoint.
      data: Any (picklable) python object containing the data to store in the
        checkpoint.
    """
    if iteration_number % self._checkpoint_frequency != 0:
      return

    filename = self._generate_filename(self._checkpoint_file_prefix,
                                       iteration_number)
    self._save_data_to_file(data, filename)
    filename = self._generate_filename('sentinel_checkpoint_complete',
                                       iteration_number)
    with tf.gfile.GFile(filename, 'wb') as fout:
      fout.write('done')

    self._clean_up_old_checkpoints(iteration_number)

  def _clean_up_old_checkpoints(self, iteration_number):
    """Removes sufficiently old checkpoints."""
    # After writing a the checkpoint and sentinel file, we garbage collect files
    # that are CHECKPOINT_DURATION * self._checkpoint_frequency versions old.
    stale_iteration_number = iteration_number - (self._checkpoint_frequency *
                                                 CHECKPOINT_DURATION)

    if stale_iteration_number >= 0:
      stale_file = self._generate_filename(self._checkpoint_file_prefix,
                                           stale_iteration_number)
      stale_sentinel = self._generate_filename('sentinel_checkpoint_complete',
                                               stale_iteration_number)
      try:
        tf.gfile.Remove(stale_file)
        tf.gfile.Remove(stale_sentinel)
      except tf.errors.NotFoundError:
        # Ignore if file not found.
        tf.logging.info('Unable to remove {} or {}.'.format(stale_file,
                                                            stale_sentinel))

  def _load_data_from_file(self, filename):
    if not tf.gfile.Exists(filename):
      return None
    with tf.gfile.GFile(filename, 'rb') as fin:
      return pickle.load(fin)

  def load_checkpoint(self, iteration_number):
    """Tries to reload a checkpoint at the selected iteration number.

    Args:
      iteration_number: The checkpoint iteration number to try to load.

    Returns:
      If the checkpoint files exist, two unpickled objects that were passed in
        as data to save_checkpoint; returns None if the files do not exist.
    """
    checkpoint_file = self._generate_filename(self._checkpoint_file_prefix,
                                              iteration_number)
    return self._load_data_from_file(checkpoint_file)

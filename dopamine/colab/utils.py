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
"""This provides utilities for dealing with Dopamine data.

SECURITY FIX: Replaced all pickle.load() calls with msgpack deserialization
to eliminate Remote Code Execution (RCE) when load_statistics() or
load_baselines() is pointed at an attacker-controlled remote path.

See: dopamine/common/logger.py .
"""

import itertools
import os
import re

import msgpack
import msgpack_numpy
import numpy as np
import pandas as pd
import tensorflow as tf


FILE_PREFIX = 'log'
ITERATION_PREFIX = 'iteration_'

ALL_GAMES = [
    'AirRaid',
    'Alien',
    'Amidar',
    'Assault',
    'Asterix',
    'Asteroids',
    'Atlantis',
    'BankHeist',
    'BattleZone',
    'BeamRider',
    'Berzerk',
    'Bowling',
    'Boxing',
    'Breakout',
    'Carnival',
    'Centipede',
    'ChopperCommand',
    'CrazyClimber',
    'DemonAttack',
    'DoubleDunk',
    'ElevatorAction',
    'Enduro',
    'FishingDerby',
    'Freeway',
    'Frostbite',
    'Gopher',
    'Gravitar',
    'Hero',
    'IceHockey',
    'Jamesbond',
    'JourneyEscape',
    'Kangaroo',
    'Krull',
    'KungFuMaster',
    'MontezumaRevenge',
    'MsPacman',
    'NameThisGame',
    'Phoenix',
    'Pitfall',
    'Pong',
    'Pooyan',
    'PrivateEye',
    'Qbert',
    'Riverraid',
    'RoadRunner',
    'Robotank',
    'Seaquest',
    'Skiing',
    'Solaris',
    'SpaceInvaders',
    'StarGunner',
    'Tennis',
    'TimePilot',
    'Tutankham',
    'UpNDown',
    'Venture',
    'VideoPinball',
    'WizardOfWor',
    'YarsRevenge',
    'Zaxxon',
]
MUJOCO_GAMES = ['Ant', 'HalfCheetah', 'Hopper', 'Humanoid', 'Walker2d']


# ---------------------------------------------------------------------------
# URI / path security helpers
# ---------------------------------------------------------------------------

_BLOCKED_REMOTE_RE = re.compile(
    r'^(?:'
    r'[a-zA-Z][a-zA-Z0-9+\-.]*://'
    r'|\\\\[^\\]'
    r')',
    re.IGNORECASE,
)


def _validate_local_path(path, label='path'):
  """Raises ValueError if path is a remote URI or UNC path.

  Args:
    path: str, the filesystem path or URI to validate.
    label: str, human-readable name used in the error message.

  Raises:
    ValueError: if the path matches a remote URI scheme or UNC path.
  """
  if _BLOCKED_REMOTE_RE.match(path):
    raise ValueError(
        'Security error: refusing to read from remote path for {}: {!r}. '
        'Only local filesystem paths are permitted. Remote paths can be '
        'used to deliver untrusted payloads resulting in Remote Code '
        'Execution.'.format(label, path)
    )


# ---------------------------------------------------------------------------
# msgpack helpers
# ---------------------------------------------------------------------------

def _unpack(raw_bytes):
  """Deserialize msgpack bytes to a Python object with numpy support."""
  return msgpack.unpackb(
      raw_bytes,
      object_hook=msgpack_numpy.decode,
      raw=False,
      strict_map_key=False,
  )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_baselines(base_dir, verbose=False):
  """Reads in the baseline experimental data from a specified base directory.

  Args:
    base_dir: string, base directory where to read data from.
      Must be a local filesystem path; remote URIs are rejected.
    verbose: bool, whether to print warning messages.

  Returns:
    A dict containing pandas DataFrames for all available agents and games.
  """
  _validate_local_path(base_dir, 'base_dir')

  experimental_data = {}
  for game in ALL_GAMES:
    for agent in ['dqn', 'c51', 'rainbow', 'iqn']:
      game_data_file = os.path.join(
          base_dir, agent, '{}.msgpack'.format(game)
      )

      if not tf.io.gfile.exists(game_data_file):
        game_data_file = os.path.join(
            base_dir, agent, '{}.pkl'.format(game)
        )

      if not tf.io.gfile.exists(game_data_file):
        if verbose:
          print(
              'Unable to load data for agent {} on game {}'.format(agent, game)
          )
        continue

      with tf.io.gfile.GFile(game_data_file, 'rb') as f:
        raw = f.read()

      single_agent_data = _unpack(raw)
      single_agent_data['agent'] = agent

      for field_name in single_agent_data.keys():
        try:
          single_agent_data[field_name] = single_agent_data[
              field_name
          ].astype(np.float64)
        except (ValueError, AttributeError):
          continue

      if game in experimental_data:
        experimental_data[game] = experimental_data[game].merge(
            single_agent_data, how='outer'
        )
      else:
        experimental_data[game] = single_agent_data

  return experimental_data


def load_statistics(log_path, iteration_number=None, verbose=True):
  """Reads in a statistics object from log_path.

  Args:
    log_path: string, full path to the training/eval statistics.
      Must be a local filesystem path; remote URIs are rejected to prevent
      loading of attacker-controlled payloads.
    iteration_number: The iteration number of the statistics object we want
      to read. If set to None, load the latest version.
    verbose: Whether to output information about the load procedure.

  Returns:
    data: The requested statistics object.
    iteration: The corresponding iteration number.

  Raises:
    ValueError: if log_path is a remote URI.
    Exception: if data is not present.
  """
  _validate_local_path(log_path, 'log_path')

  if iteration_number is None:
    iteration_number = get_latest_iteration(log_path)

  log_file = '%s/%s_%d' % (log_path, FILE_PREFIX, iteration_number)

  if verbose:
    print('Reading statistics from: {}'.format(log_file))

  with tf.io.gfile.GFile(log_file, 'rb') as f:
    data = _unpack(f.read())

  return data, iteration_number


def get_latest_file(path):
  """Return the file named 'path_[0-9]*' with the largest such number.

  Args:
    path: The base path (including directory and base name) to search.

  Returns:
    The latest file (in terms of given numbers).
  """
  try:
    latest_iteration = get_latest_iteration(path)
    return os.path.join(path, '{}_{}'.format(FILE_PREFIX, latest_iteration))
  except ValueError:
    return None


def get_latest_iteration(path):
  """Return the largest iteration number corresponding to the given path.

  Args:
    path: The base path (including directory and base name) to search.

  Returns:
    The latest iteration number.

  Raises:
    ValueError: if there is no available log data at the given path.
  """
  glob = os.path.join(path, '{}_[0-9]*'.format(FILE_PREFIX))
  log_files = tf.io.gfile.glob(glob)

  if not log_files:
    raise ValueError('No log data found at {}'.format(path))

  def extract_iteration(x):
    return int(x[x.rfind('_') + 1:])

  latest_iteration = max(extract_iteration(x) for x in log_files)
  return latest_iteration


def summarize_data(data, summary_keys):
  """Processes log data into a per-iteration summary.

  Args:
    data: Dictionary loaded by load_statistics describing the data. This
      dictionary has keys iteration_0, iteration_1, ... describing per-iteration
      data.
    summary_keys: List of per-iteration data to be summarized.

  Example:
    data = load_statistics(...)
    summarize_data(data, ['train_episode_returns', 'eval_episode_returns'])

  Returns:
    A dictionary mapping each key in returns_keys to a per-iteration summary.
  """
  summary = {}
  latest_iteration_number = len(data.keys())
  current_value = None

  for key in summary_keys:
    summary[key] = []
    for i in range(latest_iteration_number):
      iter_key = '{}{}'.format(ITERATION_PREFIX, i)
      if iter_key in data:
        current_value = np.mean(data[iter_key][key])
      summary[key].append(current_value)

  return summary


def read_experiment(
    log_path,
    parameter_set=None,
    job_descriptor='',
    iteration_number=None,
    summary_keys=('train_episode_returns', 'eval_episode_returns'),
    verbose=False,
):
  """Reads in a set of experimental results from log_path.

  Args:
    log_path: string, base path specifying where results live.
    parameter_set: An ordered_dict mapping parameter names to allowable values.
    job_descriptor: A job descriptor string used to construct the full path
      for each trial within an experiment.
    iteration_number: Int, if not None determines the iteration number at which
      we read in results.
    summary_keys: Iterable of strings, iteration statistics to summarize.
    verbose: If True, print out additional information.

  Returns:
    A Pandas dataframe containing experimental results.
  """
  keys = [] if parameter_set is None else list(parameter_set.keys())
  ordered_values = [parameter_set[key] for key in keys]

  column_names = keys + ['iteration'] + list(summary_keys)
  num_parameter_settings = len([_ for _ in itertools.product(*ordered_values)])
  expected_num_iterations = 200
  expected_num_rows = num_parameter_settings * expected_num_iterations

  data_frame = pd.DataFrame(
      index=np.arange(0, expected_num_rows), columns=column_names
  )
  row_index = 0

  for parameter_tuple in itertools.product(*ordered_values):
    if job_descriptor is not None:
      name = job_descriptor.format(*parameter_tuple)
    else:
      name = '-'.join(
          [keys[i] + '_' + str(parameter_tuple[i]) for i in range(len(keys))]
      )

    experiment_path = '{}/{}/logs'.format(log_path, name)

    raw_data, last_iteration = load_statistics(
        experiment_path, iteration_number=iteration_number, verbose=verbose
    )

    summary = summarize_data(raw_data, summary_keys)
    for iteration in range(last_iteration + 1):
      row_data = (
          list(parameter_tuple)
          + [iteration]
          + [summary[key][iteration] for key in summary_keys]
      )
      data_frame.loc[row_index] = row_data
      row_index += 1

  for field_name in data_frame.keys():
    try:
      data_frame[field_name] = data_frame[field_name].astype(np.float64)
    except ValueError:
      continue

  return data_frame.drop(np.arange(row_index, expected_num_rows))
# coding=utf-8
# Copyright 2023 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A TFDS replay with similar API as Dopamine replay buffer."""

import os

from absl import logging
from dopamine.labs.offline_rl.rlu_tfds import scaling_dataset_utils
from dopamine.labs.offline_rl.rlu_tfds import tfds_atari_utils
import gin


def get_atari_ds_name_from_replay(
    path: str, tfds_dataset: str = 'rlu_atari_checkpoints_ordered'
) -> str:
  """Returns `game_name` and `run_number` using replay directory `path`."""
  # `path` is assumed to be `.../game_name/run_number/replay_suffix`.
  replay_dir, _ = os.path.split(path)
  replay_dir, run_number = os.path.split(replay_dir)
  run_number = int(run_number)
  _, game = os.path.split(replay_dir)
  assert run_number >= 1 and run_number <= 5, 'Run number must be in [1, 5]'
  return f'{tfds_dataset}/{game}_run_{run_number}'


def game_from_dataset_name(dataset_name: str) -> str:
  return dataset_name.split('/')[1].split('_')[0]


@gin.configurable
class JaxFixedReplayBufferTFDS(object):
  """Replay Buffers for loading existing data."""

  def __init__(
      self,
      replay_capacity,
      batch_size,
      dataset_name,
      dataset_expertise=None,
      stack_size=4,
      update_horizon=1,
      gamma=0.99,
      return_to_go=False,
      **unused_kwargs,
  ):
    """Initialize the JaxMultiTaskFixedReplayBuffer class.

    Args:
      replay_capacity: int, number of transitions to keep in memory. This can be
        used with `replay_transitions_start_index` to read a subset of replay
        data starting from a specific position.
      batch_size: int, Batch size for sampling data from buffer.
      dataset_name: str, Name of the game to load data from.
      dataset_expertise: Optional[float], expertise of the dataset to use.
      stack_size: int, number of frames to use in state stack.
      update_horizon: int, length of update ('n' in n-step update).
      gamma: int, the discount factor.
      return_to_go: bool, whether to add return_to_go to the replay dataset.
      unused_kwargs: Unused kwargs arguments.
    """

    logging.info('Creating JaxFixedReplayBufferTFDS with the parameters:')
    logging.info('\t Batch size: %d', batch_size)
    logging.info('\t Update horizon: %d', update_horizon)
    logging.info('\t Gamma: %f', gamma)
    logging.info('\t Return To Go: %s', gamma)
    logging.info('\t TFDS dataset: %s', dataset_name)
    logging.info('\t Dataset expertise: %s', dataset_expertise)
    # data % = (replay_capacity / 1M) x 100
    self._data_percent = float(replay_capacity) / 10_000.0
    self._replay_capacity = replay_capacity
    self._batch_size = batch_size
    self._return_to_go = return_to_go
    self._total_steps = self._data_percent * 50_000_000
    self._dataset_expertise = dataset_expertise
    if dataset_expertise is not None:
      dataset, (self._total_steps, self._min_score, self._max_score) = (
          scaling_dataset_utils.create_dataset_with_expertise(
              game=game_from_dataset_name(dataset_name),
              dataset_expertise=dataset_expertise,
          )
      )
    else:
      dataset = tfds_atari_utils.uniformly_subsampled_atari_data(
          dataset_name, self._data_percent
      )
    logging.info('\t StepsPerEpoch %d', self.gradient_steps_per_epoch)
    self._replay = tfds_atari_utils.build_tfds_replay(
        dataset=dataset,
        stack_size=stack_size,
        update_horizon=update_horizon,
        gamma=gamma,
        batch_size=batch_size,
        return_to_go=return_to_go,
    )

  @property
  def min_max_returns(self):
    """Returns min and max clipped returns."""
    if self._dataset_expertise is not None:
      return (self._min_score, self._max_score)
    raise NotImplementedError

  def _load_buffer(self, suffix):
    """Not needed with tfds datasets."""
    pass

  def load_single_buffer(self, suffix):
    pass

  def _load_replay_buffers(self, unused_num_buffers):
    pass

  def get_transition_elements(self):
    raise NotImplementedError

  def sample_transition_batch(self):
    """Sample a transition batch from the tfds."""
    return next(self._replay)

  def load(self, *args, **kwargs):  # pylint: disable=unused-argument
    pass

  def reload_buffer(self, num_buffers):
    pass

  def save(self, *args, **kwargs):  # pylint: disable=unused-argument
    pass

  def add(self, *args, **kwargs):  # pylint: disable=unused-argument
    pass

  @property
  def add_count(self):
    return self._total_steps

  @property
  def gradient_steps_per_epoch(self):
    return self._total_steps // self._batch_size

  @property
  def replay_capacity(self):
    return self._replay_capacity

  def reload_data(self):
    pass

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
"""Helpers for using RLU Atari datasets for offline RL."""

import rlds
import tensorflow as tf
import tensorflow_datasets as tfds


class BatchToTransition(object):
  """Creates (s,a,r,s',a') transitions."""

  def __init__(self, stack_size, update_horizon, gamma,
               next_action=False,
               return_to_go=False):
    self.stack_size = stack_size
    self.update_horizon = update_horizon
    self.total_frames = stack_size + update_horizon
    self.cumulative_discount = tf.pow(gamma, range(update_horizon))
    self.next_action = next_action
    self.return_to_go = return_to_go

  def create_transitions(self, batch, episode_return=None):
    """Convert a batch to transition tuple."""
    all_states = tf.squeeze(batch[rlds.OBSERVATION], axis=-1)
    all_states = tf.transpose(all_states, perm=[1, 2, 0])
    rewards = batch[rlds.REWARD][self.stack_size-1:-1]
    terminals = batch[rlds.IS_TERMINAL][self.stack_size: self.total_frames]
    transitions = {
        'state': all_states[:, :, :self.stack_size],
        'action': batch[rlds.ACTION][self.stack_size-1],
        'reward': tf.reduce_sum(rewards * self.cumulative_discount),
        'next_state': all_states[:, :, self.update_horizon:],
        'terminal': tf.reduce_any(terminals),
    }
    if episode_return is not None:
      transitions['episode_return'] = episode_return
    if self.next_action:
      transitions['next_action'] = batch[rlds.ACTION][self.total_frames - 1]
    if self.return_to_go:
      transitions['return_to_go'] = batch['return_to_go'][self.stack_size - 1]
    return transitions


def get_transition_dataset_fn(stack_size, update_horizon=1, gamma=0.99,
                              return_to_go=False):
  """Creates a dataset of (s, a, r, s', a') transitions."""
  batch_fn_cls = BatchToTransition(
      stack_size, update_horizon, gamma, return_to_go=return_to_go)

  def _add_rtg(episode_data):
    episode_data['return_to_go'] = tf.cumsum(
        episode_data[rlds.REWARD], reverse=True)
    return episode_data

  def make_transition_dataset(episode_data):
    """Converts an episode of steps to a dataset of custom transitions."""
    # Create a dataset of 2-step sequences with overlap of 1.
    episode = episode_data[rlds.STEPS]
    if return_to_go:
      episode = episode.map(_add_rtg)
    batched_steps = rlds.transformations.batch(
        episode,
        size=stack_size + update_horizon,
        shift=1,
        drop_remainder=True)
    # pylint: disable=g-long-lambda
    batch_fn = lambda x: batch_fn_cls.create_transitions(
        x, episode_return=episode_data['episode_return'])
    # pylint: enable=g-long-lambda
    return batched_steps.map(batch_fn,
                             num_parallel_calls=tf.data.AUTOTUNE)
  return make_transition_dataset


def load_data_splits(dataset_name, data_splits):
  # Interleave episodes across different splits/checkpoints
  # Set `shuffle_files=True` to shuffle episodes across files within splits
  read_config = tfds.ReadConfig(
      interleave_cycle_length=len(data_splits),
      shuffle_reshuffle_each_iteration=True,
      enable_ordering_guard=False,
  )
  return tfds.load(
      dataset_name, split='+'.join(data_splits),
      read_config=read_config,
      shuffle_files=True,
  )


def uniformly_subsampled_atari_data(dataset_name, data_percent):
  """Load `data_percent` sampled from rlu tfds `dataset_name` dataset."""
  ds_builder = tfds.builder(dataset_name)
  data_splits = []
  for split, info in ds_builder.info.splits.items():
    # Convert `data_percent` to number of episodes to allow
    # for fractional percentages.
    num_episodes = int((data_percent/100) * info.num_examples)
    if num_episodes == 0:
      raise ValueError(f'{data_percent}% leads to 0 episodes in {split}!')
    # Sample first `data_percent` episodes from each of the data split
    data_splits.append(f'{split}[:{num_episodes}]')
  return load_data_splits(dataset_name, data_splits)


def create_atari_ds_loader(dataset,
                           transition_fn=None,
                           shuffle_num_episodes=1000,
                           shuffle_num_steps=50000,
                           cycle_length=100):
  """Create uniformly subsampled Atari `game` dataset."""
  if transition_fn is None:
    transition_fn = get_transition_dataset_fn(4)
  # Shuffle the episodes to avoid consecutive episodes
  dataset = dataset.shuffle(shuffle_num_episodes)
  # Interleave the steps across many different episodes
  dataset = dataset.interleave(
      transition_fn, cycle_length=cycle_length, block_length=1,
      deterministic=False, num_parallel_calls=tf.data.AUTOTUNE)
  # Shuffle steps in the dataset
  shuffled_dataset = dataset.shuffle(
      shuffle_num_steps, reshuffle_each_iteration=True)
  return shuffled_dataset


def create_ds_iterator(ds, batch_size=32, repeat=True):
  """Create numpy iterator from a tf dataset `ds`."""
  if repeat:
    ds = ds.repeat()
  batch_ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
  return batch_ds.as_numpy_iterator()


def build_tfds_replay(stack_size,
                      update_horizon,
                      gamma,
                      return_to_go=False,
                      dataset=None,
                      game='Pong',
                      run_number=1,
                      batch_size=32,
                      repeat_ds=True):
  """Builds the tfds replay buffer."""
  transition_fn = get_transition_dataset_fn(
      stack_size, update_horizon, gamma, return_to_go=return_to_go)
  if dataset is None:
    dataset_name = f'rlu_atari_checkpoints_ordered/{game}_run_{run_number}'
    # Create a dataset of episodes sampling `data_percent`% episodes
    # from each of the data split
    dataset = uniformly_subsampled_atari_data(dataset_name, data_percent=10)
  atari_ds = create_atari_ds_loader(dataset, transition_fn=transition_fn)
  return create_ds_iterator(atari_ds, batch_size, repeat=repeat_ds)

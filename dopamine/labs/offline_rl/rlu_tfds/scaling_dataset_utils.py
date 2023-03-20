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
"""Utils for creating datasets for scaling offline RL."""

import concurrent.futures

from dopamine.labs.offline_rl.rlu_tfds import tfds_atari_utils
import gin
import numpy as np
import tensorflow_datasets as tfds


BatchToTransition = tfds_atari_utils.BatchToTransition
get_transition_dataset_fn = tfds_atari_utils.get_transition_dataset_fn
load_data_splits = tfds_atari_utils.load_data_splits


def _parallel_lookup(game, episodes_per_policy, max_checkpoints=50,
                     num_datasets=5):
  """Get idxs to use for dataset splits to be loaded."""

  def _get_lookups(i, j, ds_chkpt, episodes_per_policy):
    policy_value_lookup = {}
    policy_ep_idx_lookup = {}
    ds = ds_chkpt.map(lambda x: (x['episode_return'], len(x['steps'])))
    info = list(ds.as_numpy_iterator())
    policy_id = (i, j)
    episode_returns = [ep_return for ep_return, _ in info]
    episode_lengths = [num_steps for _, num_steps in info]
    if len(info) < episodes_per_policy:
      print(
          f'Warning! Not enough episodes in this chunk ({len(info)}) to'
          f' support {episodes_per_policy} episodes per policy.'
      )
    policy_value_lookup[policy_id] = np.mean(episode_returns)
    policy_ep_idx_lookup[policy_id] = (
        len(info), episode_lengths, episode_returns)
    return policy_value_lookup, policy_ep_idx_lookup

  policy_value_lookup, policy_ep_idx_lookup = {}, {}
  with concurrent.futures.ThreadPoolExecutor() as executor:
    # Start the load operations and mark each future with its URL
    jobs = []
    for i in range(1, num_datasets + 1):
      dataset_name = f'rlu_atari_checkpoints_ordered/{game}_run_{i}'
      ds = tfds.load(dataset_name)
      for j, chkpt in enumerate(ds):
        # Use only the first 25 ckpts.
        if int(chkpt.split('_')[1]) < max_checkpoints:
          job = executor.submit(
              _get_lookups, i, j, ds[chkpt], episodes_per_policy)
          jobs.append(job)
    for future in concurrent.futures.as_completed(jobs):
      try:
        pval, ep_idx = future.result()
        policy_value_lookup.update(pval)
        policy_ep_idx_lookup.update(ep_idx)
      except Exception as exc:  # pylint: disable=broad-except
        print(f'Generated an exception: {exc}')
  return policy_value_lookup, policy_ep_idx_lookup


def choose_indices(
    game,
    dataset_expertise,
    dataset_policies,
    num_datasets=5,
    dataset_episodes=10000,
    returns_filter=None,
):
  """Create indices to be used for dataset creation."""
  episodes_per_policy = dataset_episodes // dataset_policies
  (policy_value_lookup, policy_ep_idx_lookup) = _parallel_lookup(
      game, episodes_per_policy, num_datasets=num_datasets,
      max_checkpoints=dataset_policies)
  policy_ids = list(policy_value_lookup.keys())
  num_ids = len(policy_ids)
  sorted_policy_ids = list(
      sorted(policy_ids, key=lambda pid: policy_value_lookup[pid])
  )
  inexpert_policy_ids = sorted_policy_ids[: int(dataset_expertise * num_ids)]
  if returns_filter is not None:
    inexpert_policy_ids = inexpert_policy_ids[
        int(returns_filter * len(inexpert_policy_ids)) :]

  if len(inexpert_policy_ids) < dataset_policies:
    selected_policy_ids = inexpert_policy_ids
  else:
    selected_policy_ids = [
        inexpert_policy_ids[int(i)] for i in
        np.linspace(0, len(inexpert_policy_ids)-1, dataset_policies)
    ]
  min_score, max_score = float('inf'), float('-inf')
  total_steps = 0

  all_episode_idxs = []
  for policy_id in selected_policy_ids:
    l, ep_lengths, ep_returns = policy_ep_idx_lookup[policy_id]
    x1 = episodes_per_policy // 2
    start_idx, end_idx = max(l // 2 - x1, 0), min(l//2 + x1, l)
    all_episode_idxs.append((
        policy_id[0], policy_id[1], f'{start_idx}:{end_idx}'))
    # Book keeping
    total_steps += sum(ep_lengths[start_idx:end_idx])
    for ep_return in ep_returns[start_idx:end_idx]:
      min_score = min(min_score, ep_return)
      max_score = max(max_score, ep_return)
  return all_episode_idxs, (total_steps, min_score, max_score)


def _get_dataset_with_idxs(game, idxs, num_datasets):
  """Create a game dataset using splits based on idxs."""

  def _generate_data_splits(dataset_name, idxs):
    ds_builder = tfds.builder(dataset_name)
    split_lookup = list(ds_builder.info.splits.keys())
    data_splits = [f'{split_lookup[j]}[{k}]' for _, j, k in idxs]
    if not data_splits:
      return None
    return load_data_splits(dataset_name, data_splits)

  full_dataset = None
  for dataset_i in range(1, num_datasets + 1):
    # Create a dataset of episodes
    filtered_idxs = [idx for idx in idxs if idx[0] == dataset_i]
    dataset_name = f'rlu_atari_checkpoints_ordered/{game}_run_{dataset_i}'
    dataset = _generate_data_splits(dataset_name, list(filtered_idxs))
    if dataset is None:
      continue
    if full_dataset is None:
      full_dataset = dataset
    else:
      full_dataset = full_dataset.concatenate(dataset)
  return full_dataset


@gin.configurable
def create_dataset_with_expertise(
    game,
    dataset_expertise=1.0,
    dataset_policies=40,
    dataset_episodes=10000,
    num_datasets=5,
):
  """Create offline dataset with varying expertise."""
  idxs, aux_outputs = choose_indices(
      game,
      dataset_expertise=dataset_expertise,
      dataset_policies=dataset_policies,
      dataset_episodes=dataset_episodes,
      num_datasets=num_datasets,
  )
  dataset = _get_dataset_with_idxs(game, idxs, num_datasets=num_datasets)
  return dataset, aux_outputs

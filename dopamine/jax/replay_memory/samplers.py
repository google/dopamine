# coding=utf-8
# Copyright 2024 The Dopamine Authors.
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
"""Sampling distributions."""

import dataclasses
import typing
from typing import Any, Protocol

from absl import logging
from dopamine.jax import checkpointers
from dopamine.jax.replay_memory import elements
from dopamine.jax.replay_memory import sum_tree
import gin
import numpy as np
import numpy.typing as npt

ReplayItemID = elements.ReplayItemID


@dataclasses.dataclass(frozen=True, kw_only=True)
class SampleMetadata:
  keys: npt.NDArray[ReplayItemID]


@typing.runtime_checkable
class SamplingDistribution(checkpointers.Checkpointable, Protocol):
  """A sampling distribution."""

  def add(self, key: ReplayItemID, **kwargs: Any) -> None:
    ...

  @typing.overload
  def update(self, keys: npt.NDArray[ReplayItemID], **kwargs: Any) -> None:
    ...

  @typing.overload
  def update(self, key: ReplayItemID, **kwargs: Any) -> None:
    ...

  def update(
      self, keys: npt.NDArray[ReplayItemID] | ReplayItemID, **kwargs: Any
  ) -> None:
    ...

  def remove(self, key: ReplayItemID) -> None:
    ...

  def sample(self, size: int) -> SampleMetadata:
    ...

  def clear(self) -> None:
    ...


class UniformSamplingDistribution(SamplingDistribution):
  """A uniform sampling distribution."""

  def __init__(
      self, seed: np.random.Generator | np.random.SeedSequence | int | None
  ) -> None:
    # RNG generator
    self._rng = np.random.default_rng(seed)

    # The following two datastructures are used for efficient sampling from
    # our memory collection. It's not possible with Python only to efficiently
    # sample from a dictionary without incurring an O(n) cost to materialize
    # the keys / values.
    # To efficiently sample from the dictionary we maintain two datastructures:
    #   1) A mapping (as a dict) from dict keys -> contiguous local indices
    #   2) A mapping (as a list) from local indices -> dict keys
    #
    # Now to sample from our dictionary we can sample from the range of
    # local indices (of which we know the bounds for). Then we can fetch
    # the dictionary keys associated with those indices to then lookup in our
    # dictionary. The cost is now amortized constant across both operations.
    #
    # See `replay_buffer_test.py:testKeyMappingsForSampling` for an example.

    # A mapping (as a dict) from dict keys -> local indices
    # e.g., consider a replay buffer with max size 100, the global key 100 would
    # map to a local index of 0 (wrap around).
    self._index_by_key = {}
    # A reverse mapping (as an array) from local indices -> global keys
    self._key_by_index = []

  @property
  def size(self) -> int:
    return len(self._key_by_index)

  def add(self, key: ReplayItemID, **kwargs: Any) -> None:
    if kwargs:
      logging.log_first_n(
          logging.WARN,
          'Got kwargs %r to `UniformDistribution.add`, ignoring...',
          1,
          kwargs,
      )
    # Keep track of the index of this key
    self._index_by_key[key] = len(self._key_by_index)
    self._key_by_index.append(key)

  @typing.overload
  def update(
      self, keys: npt.NDArray[ReplayItemID], *args: Any, **kwargs: Any
  ) -> None:
    ...

  @typing.overload
  def update(self, key: ReplayItemID, *args: Any, **kwargs: Any) -> None:
    ...

  def update(
      self,
      keys: npt.NDArray[ReplayItemID] | ReplayItemID,
      *args: Any,
      **kwargs: Any,
  ) -> None:
    logging.log_first_n(
        logging.WARN, '`UniformDistribution.update` is a no-op, ignoring...', 1
    )

  def remove(self, key: ReplayItemID) -> None:
    if key not in self._index_by_key:
      raise ValueError(f'Key {key} not found.')
    # We now must resolve the index from this key to update our map
    index = self._index_by_key[key]
    # Swap the values of the oldest key and the latest key
    # so we can perform efficient O(1) pop on the keys
    self._key_by_index[index], self._key_by_index[-1] = (
        self._key_by_index[-1],
        self._key_by_index[index],
    )
    # Make sure we update the key->index mapping with the swapped index
    self._index_by_key[self._key_by_index[index]] = index
    # Pop the oldest key to get the index then pop the index from the dict
    self._index_by_key.pop(self._key_by_index.pop())

  def sample(self, size: int) -> SampleMetadata:
    if not self._key_by_index:
      raise ValueError('No keys to sample from.')
    if size <= 0:
      raise ValueError('Sample size must be positive.')
    indices = self._rng.integers(len(self._key_by_index), size=size)
    return SampleMetadata(
        keys=np.fromiter(
            (self._key_by_index[index] for index in indices),
            dtype=np.int32,
            count=size,
        )
    )

  def clear(self) -> None:
    self._index_by_key.clear()
    self._key_by_index.clear()

  def to_state_dict(self) -> dict[str, Any]:
    return {
        'key_by_index': self._key_by_index,
        'index_by_key': self._index_by_key,
        'rng_state': self._rng.bit_generator.state,
    }

  def from_state_dict(self, state_dict: dict[str, Any]) -> None:
    self._key_by_index = state_dict['key_by_index']
    self._index_by_key = state_dict['index_by_key']

    # Restore rng state
    self._rng.bit_generator.state = state_dict['rng_state']


@dataclasses.dataclass(frozen=True, kw_only=True)
class PrioritizedSampleMetadata(SampleMetadata):
  probabilities: npt.NDArray[np.float64]


@gin.configurable
class PrioritizedSamplingDistribution(UniformSamplingDistribution):
  """A prioritized sampling distribution."""

  def __init__(
      self,
      seed: np.random.SeedSequence | int | None,
      *,
      priority_exponent: float = 1.0,
      max_capacity: int,
  ) -> None:
    self._rng = np.random.default_rng(seed)
    self._max_capacity = max_capacity
    self._priority_exponent = priority_exponent
    self._sum_tree = sum_tree.SumTree(self._max_capacity)

    super().__init__(self._rng)

  def add(self, key: ReplayItemID, *, priority: float) -> None:
    super().add(key)
    if priority is None:
      priority = 0.0
    self._sum_tree.set(
        self._index_by_key[key],
        0.0 if priority == 0.0 else priority**self._priority_exponent,
    )

  @typing.overload
  def update(
      self,
      keys: npt.NDArray[ReplayItemID],
      *,
      priorities: npt.NDArray[np.float64],
  ) -> None:
    ...

  @typing.overload
  def update(self, keys: ReplayItemID, *, priorities: float) -> None:
    ...

  def update(
      self,
      keys: npt.NDArray[ReplayItemID] | ReplayItemID,
      *,
      priorities: npt.NDArray[np.float64] | float,
  ) -> None:
    if not isinstance(keys, np.ndarray):
      keys = np.asarray([keys], dtype=np.int32)

    priorities = np.where(
        priorities == 0.0, 0.0, priorities**self._priority_exponent
    )
    self._sum_tree.set(
        np.fromiter((self._index_by_key[key] for key in keys), dtype=np.int32),
        priorities,
    )

  def remove(self, key: ReplayItemID) -> None:
    index = self._index_by_key[key]
    last_index = len(self._key_by_index) - 1
    if index == last_index:
      # If index and last_index are the same, simply set the priority to 0.0.
      self._sum_tree.set(index, 0.0)
    else:
      # Otherwise, swap priorities with current index and last index
      # as that's how we pop the key from our datastructure.
      # This will run in O(logn) where n is the # of elements in the tree
      self._sum_tree.set(
          np.asarray([index, last_index], dtype=np.int32),
          np.asarray([self._sum_tree.get(last_index), 0.0]),
      )
    super().remove(key)

  def sample(self, size: int) -> PrioritizedSampleMetadata:
    if self._sum_tree.root == 0.0:
      keys = super().sample(size).keys
      return PrioritizedSampleMetadata(
          keys=keys,
          probabilities=np.full_like(keys, 1.0 / self.size, dtype=np.float64),
      )

    targets = self._rng.uniform(0.0, self._sum_tree.root, size=size)
    indices = self._sum_tree.query(targets)
    return PrioritizedSampleMetadata(
        keys=np.fromiter(
            (self._key_by_index[index] for index in indices),
            count=size,
            dtype=np.int32,
        ),
        probabilities=self._sum_tree.get(indices) / self._sum_tree.root,
    )

  def clear(self) -> None:
    self._sum_tree.clear()
    super().clear()

  def to_state_dict(self) -> dict[str, Any]:
    return {
        'sum_tree': self._sum_tree.to_state_dict(),
        **super().to_state_dict(),
    }

  def from_state_dict(self, state_dict: dict[str, Any]):
    super().from_state_dict(state_dict)
    self._sum_tree.from_state_dict(state_dict['sum_tree'])


class SequentialSamplingDistribution(UniformSamplingDistribution):
  """A sequential sampling distribution."""

  def __init__(
      self,
      seed: np.random.Generator | np.random.SeedSequence | int,
      sort_samples: bool = True,
  ):
    super().__init__(seed)
    self._sort_samples = sort_samples

  def sample(self, size: int) -> SampleMetadata:
    if not self._key_by_index:
      raise ValueError('No keys to sample from.')
    if size <= 0:
      raise ValueError('Sample size must be positive.')
    if self._sort_samples:
      self._key_by_index.sort()
    return SampleMetadata(
        keys=np.fromiter(
            self._key_by_index,
            dtype=np.int32,
            count=size,
        )
    )

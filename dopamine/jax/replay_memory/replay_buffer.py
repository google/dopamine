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
"""Simpler implementation of the standard DQN replay memory."""
import collections
import functools
import operator
import pickle
import typing
from typing import Any, Generic, Literal, TypeVar

from dopamine.jax import checkpointers
from dopamine.jax.replay_memory import accumulator
from dopamine.jax.replay_memory import elements
from dopamine.jax.replay_memory import samplers
import gin
import jax
import numpy as np
import numpy.typing as npt
from orbax import checkpoint as orbax

ReplayItemID = elements.ReplayItemID
ReplayElementT = TypeVar('ReplayElementT', bound=elements.ReplayElementProtocol)


@gin.configurable
class ReplayBuffer(checkpointers.Checkpointable, Generic[ReplayElementT]):
  """A Jax re-implementation of the Dopamine Replay Buffer.

  Stores transitions, state, action, reward, next_state, terminal (and any
  extra contents specified) in a circular buffer and provides a uniform
  sampling mechanism.

  The main changes from the original Dopamine implementation are:
  - The original Dopamine replay buffer stored the raw observations and stacked
    them upon sampling. Although space efficient, this is time inefficient. We
    use the same compression mechanism used by dqn_zoo to store only stacked
    states in a space efficient manner.
  - Similarly, n-step transitions were computed at sample time. We only store
    n-step returns in the buffer by using an Accumulator (which is also used for
    stacking).
  - The above two points allow us to sample uniformly from the FIFO queue,
    without needing to determine invalid ranges (as in the original replay
    buffer). The computation of invalid ranges was inefficient, as it required
    verifying each sampled index, and potentially resampling them if they fell
    in an invalid range. This computation scales linearly with the batch size,
    and is thus quite inefficient.
  - The original Dopamine replay buffer maintained a static array and performed
    modulo arithmetic when adding/sampling. This is unnecessarily complicated
    and we can achieve the same result by maintaining a FIFO queue (using an
    OrderedDict structure) containing only valid transitions.
  """

  def __init__(
      self,
      transition_accumulator: accumulator.Accumulator[ReplayElementT],
      sampling_distribution: samplers.SamplingDistribution,
      *,
      batch_size: int,
      max_capacity: int,
      checkpoint_duration: int = 4,
      compress: bool = True,
  ):
    """Initializes the ReplayBuffer."""
    self.add_count = 0
    self._max_capacity = max_capacity
    self._compress = compress
    self._memory = collections.OrderedDict[ReplayItemID, ReplayElementT]()

    self._transition_accumulator = transition_accumulator
    self._sampling_distribution = sampling_distribution

    if checkpoint_duration < 1:
      raise ValueError(f'Invalid checkpoint_duration: {checkpoint_duration}')

    self._checkpoint_duration = checkpoint_duration
    self._batch_size = batch_size

  def add(self, transition: elements.TransitionElement, **kwargs: Any) -> None:
    """Add a transition to the replay buffer."""
    for replay_element in self._transition_accumulator.accumulate(transition):
      if self._compress:
        replay_element = replay_element.pack()

      # Add replay element to memory
      key = ReplayItemID(self.add_count)
      self._memory[key] = replay_element
      self._sampling_distribution.add(key, **kwargs)
      self.add_count += 1
      # If we're beyond our capacity...
      if self.add_count > self._max_capacity:
        # Pop the oldest item from memory and keep the key
        # so we can ask the sampling distribution to remove it
        oldest_key, _ = self._memory.popitem(last=False)
        self._sampling_distribution.remove(oldest_key)

  @typing.overload
  def sample(
      self,
      size: int | None = None,
      *,
      with_sample_metadata: Literal[False] = False,
  ) -> ReplayElementT:
    ...

  @typing.overload
  def sample(
      self,
      size: int,
      *,
      with_sample_metadata: Literal[True] = True,
  ) -> tuple[ReplayElementT, samplers.SampleMetadata]:
    ...

  def sample(
      self,
      size: int | None = None,
      *,
      with_sample_metadata: bool = False,
  ) -> ReplayElementT | tuple[ReplayElementT, samplers.SampleMetadata]:
    """Sample a batch of elements from the replay buffer."""
    if self.add_count < 1:
      raise ValueError('No samples in replay buffer!')
    if size is None:
      size = self._batch_size
    if size < 1:
      raise ValueError(f'Invalid size: {size}, size must be >= 1.')

    samples = self._sampling_distribution.sample(size)
    replay_elements = operator.itemgetter(*samples.keys)(self._memory)
    if not isinstance(replay_elements, tuple):
      # When size == 1, replay_elements will hold a single ReplayElement, as
      # opposed to a tuple of ReplayElements (which is what is expected in the
      # call to tree_map below). We fix this by forcing replay_elements into a
      # tuple, if necessary.
      replay_elements = (replay_elements,)
    if self._compress:
      replay_elements = map(operator.methodcaller('unpack'), replay_elements)

    batch = jax.tree_util.tree_map(lambda *xs: np.stack(xs), *replay_elements)
    return (batch, samples) if with_sample_metadata else batch

  @typing.overload
  def update(self, keys: npt.NDArray[ReplayItemID], **kwargs: Any) -> None:
    ...

  @typing.overload
  def update(self, keys: ReplayItemID, **kwargs: Any) -> None:
    ...

  def update(
      self,
      keys: npt.NDArray[ReplayItemID] | ReplayItemID,
      **kwargs: Any,
  ) -> None:
    self._sampling_distribution.update(keys, **kwargs)

  def clear(self) -> None:
    """Clear the replay buffer."""
    self.add_count = 0
    self._memory.clear()
    self._transition_accumulator.clear()
    self._sampling_distribution.clear()

  def to_state_dict(self) -> dict[str, Any]:
    """Serialize replay buffer to a state dictionary."""
    # Serialize memory. We'll serialize keys and values separately.
    keys = list(self._memory.keys())
    # To serialize values we'll flatten each transition element.
    # This will serialize replay elements as:
    #   [[state, action, reward, next_state, is_terminal, episode_end], ...]
    values = iter(self._memory.values())
    leaves, treedef = jax.tree_util.tree_flatten(next(values, None))
    values = [] if not leaves else [leaves, *map(treedef.flatten_up_to, values)]

    return {
        'add_count': self.add_count,
        'memory': {
            'keys': keys,
            'values': values,
            'treedef': pickle.dumps(treedef),
        },
        'sampling_distribution': self._sampling_distribution.to_state_dict(),
        'transition_accumulator': self._transition_accumulator.to_state_dict(),
    }

  def from_state_dict(self, state_dict: dict[str, Any]) -> None:
    """Deserialize and mutate replay buffer using state dictionary."""
    self.add_count = state_dict['add_count']
    self._transition_accumulator.from_state_dict(
        state_dict['transition_accumulator']
    )
    self._sampling_distribution.from_state_dict(
        state_dict['sampling_distribution']
    )

    # Restore memory
    memory_keys = state_dict['memory']['keys']
    # Each element of the list is a flattened replay element, unflatten them
    # i.e., we have storage like:
    #   [[state, action, reward, next_state, is_terminal, episode_end], ...]
    # and after unflattening we'll have:
    #   [ReplayElementT(...), ...]
    memory_treedef: jax.tree_util.PyTreeDef = pickle.loads(
        state_dict['memory']['treedef']
    )
    memory_values = map(
        memory_treedef.unflatten, state_dict['memory']['values']
    )

    # Create our new ordered dictionary from the restored keys and values
    self._memory = collections.OrderedDict[ReplayItemID, ReplayElementT](
        zip(memory_keys, memory_values, strict=True)
    )

  @functools.lru_cache
  def _make_checkpoint_manager(
      self, checkpoint_dir: str
  ) -> orbax.CheckpointManager:
    """Create orbax checkpoint manager, cache the manager based on path."""
    return orbax.CheckpointManager(
        checkpoint_dir,
        checkpointers={
            'replay': orbax.Checkpointer(
                checkpointers.CheckpointHandler[ReplayBuffer](),
            )
        },
        options=orbax.CheckpointManagerOptions(
            max_to_keep=self._checkpoint_duration,
            create=True,
        ),
    )

  def save(self, checkpoint_dir: str, iteration_number: int):
    """Save the ReplayBuffer attributes into a file.

    Args:
      checkpoint_dir: the directory where numpy checkpoint files should be
        saved. Must already exist.
      iteration_number: iteration_number to use as a suffix in naming.
    """
    checkpoint_manager = self._make_checkpoint_manager(checkpoint_dir)
    checkpoint_manager.save(iteration_number, {'replay': self})

  def load(self, checkpoint_dir: str, iteration_number: int):
    """Restores from a checkpoint.

    Args:
      checkpoint_dir: the directory where to read the checkpoint.
      iteration_number: iteration_number to use as a suffix in naming.
    """
    checkpoint_manager = self._make_checkpoint_manager(checkpoint_dir)
    # NOTE: Make sure not to pass in `items={'replay': self}` as this will
    # create a deep copy and we want to mutate in-place.
    # If we don't pass items then we get back a state dictionary
    # that we can use to mutate in-place.
    state_dict = checkpoint_manager.restore(iteration_number)
    self.from_state_dict(state_dict['replay'])

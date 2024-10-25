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
"""An accumulator used by the Jax replay buffer."""
import collections
import pickle
import typing
from typing import Any, Iterable, Protocol
from typing import TypeVar

from dopamine.jax import checkpointers
from dopamine.jax.replay_memory import elements
import jax
import numpy as np


ReplayElementT = TypeVar('ReplayElementT', bound=elements.ReplayElementProtocol)


@typing.runtime_checkable
class Accumulator(checkpointers.Checkpointable, Protocol[ReplayElementT]):

  def accumulate(
      self, transition: elements.TransitionElement
  ) -> Iterable[ReplayElementT]:
    ...

  def clear(self) -> None:
    ...


class TransitionAccumulator(Accumulator[elements.ReplayElement]):
  """A transition accumulator used for preparing elements for replay.

  This class will consume raw transitions and prepare them for storing in the
  replay buffer. Specifically, it will only return validly stacked frames, and
  will only return valid n-step returns.
  This enables us to guarantee that the replay buffer only contains valid
  elements to sample.
  """

  def __init__(
      self,
      stack_size: int,
      update_horizon: int,
      gamma: float,
  ):
    assert update_horizon > 0, 'update_horizon must be positive.'
    assert stack_size > 0, 'stack_size must be positive.'

    self._stack_size = stack_size
    self._update_horizon = update_horizon
    self._gamma = gamma

    self._trajectory = collections.deque[elements.TransitionElement](
        maxlen=self._update_horizon + self._stack_size
    )

  def _make_replay_element(self) -> elements.ReplayElement | None:
    trajectory_len = len(self._trajectory)

    last_transition = self._trajectory[-1]
    # Check if we have a valid transition, i.e. we either
    #   1) have accumulated more transitions than the update horizon
    #   2) have a trajectory shorter than the update horizon, but the
    #      last element is terminal
    if not (
        trajectory_len > self._update_horizon
        or (trajectory_len > 1 and last_transition.is_terminal)
    ):
      return None

    # Calculate effective horizon, this can differ from the update horizon
    # when we have n-step transitions where the last observation is terminal.
    effective_horizon = self._update_horizon
    if last_transition.is_terminal and trajectory_len <= self._update_horizon:
      effective_horizon = trajectory_len - 1

    # pytype: disable=attribute-error
    observation_shape = last_transition.observation.shape + (self._stack_size,)
    observation_dtype = last_transition.observation.dtype
    # pytype: enable=attribute-error

    o_tm1 = np.zeros(observation_shape, observation_dtype)
    # Initialize the slice for which this observation is valid.
    # The start index for o_tm1 is the start of the n-step trajectory.
    # The end index for o_tm1 is just moving over `stack size`.
    o_tm1_slice = slice(
        trajectory_len - effective_horizon - self._stack_size,
        trajectory_len - effective_horizon - 1,
    )
    # The action chosen will be the last transition in the stack.
    a_tm1 = self._trajectory[o_tm1_slice.stop].action

    o_t = np.zeros(observation_shape, observation_dtype)
    # Initialize the slice for which this observation is valid.
    # The start index for o_t is just moving backwards `stack size`.
    # The end index for o_t is just the last index of the n-step trajectory.
    o_t_slice = slice(
        trajectory_len - self._stack_size,
        trajectory_len - 1,
    )
    # Terminal information will come from the last transition in the stack
    is_terminal = self._trajectory[o_t_slice.stop].is_terminal
    episode_end = self._trajectory[o_t_slice.stop].is_terminal

    # Slice to accumulate n-step returns. This will be the end
    # transition of o_tm1 plus the effective horizon.
    # This might over-run the trajectory length in the case of n-step
    # returns where the last transition is terminal.
    gamma_slice = slice(
        o_tm1_slice.stop,
        o_tm1_slice.stop + self._update_horizon - 1,
    )
    assert o_t_slice.stop - o_tm1_slice.stop == effective_horizon
    assert o_t_slice.stop - 1 >= o_tm1_slice.stop

    # Now we'll iterate through the n-step trajectory and compute the
    # cumulant and insert the observations into the appropriate stacks
    r_t = 0.0
    for t, transition_t in enumerate(self._trajectory):
      # If we should be accumulating reward for an n-step return?
      if gamma_slice.start <= t <= gamma_slice.stop:
        r_t += transition_t.reward * (self._gamma ** (t - gamma_slice.start))

      # If we should be accumulating frames for the frame-stack?
      if o_tm1_slice.start <= t <= o_tm1_slice.stop:
        o_tm1[..., t - o_tm1_slice.start] = transition_t.observation
      if o_t_slice.start <= t <= o_t_slice.stop:
        o_t[..., t - o_t_slice.start] = transition_t.observation

    return elements.ReplayElement(
        state=o_tm1,
        action=a_tm1,
        reward=r_t,
        next_state=o_t,
        is_terminal=is_terminal,
        episode_end=episode_end,
    )

  def accumulate(
      self, transition: elements.TransitionElement
  ) -> Iterable[elements.ReplayElement]:
    """Add a transition to the accumulator, maybe receive valid ReplayElements.

    If the transition has a terminal or end of episode signal, it will create a
    new trajectory.

    If the transition is terminal the iterator will yield multiple elements
    corresponding to n-step returns with n ∊ [1, n].

    Args:
      transition: TransitionElement to add.

    Yields:
      A ReplayElement if there is a valid transition.
    """
    self._trajectory.append(transition)

    # If this transition terminated a trajectory we'll yield
    # as many replay elements as we can before clearing the trajectory
    if transition.is_terminal:
      while replay_element := self._make_replay_element():
        yield replay_element
        self._trajectory.popleft()
      self._trajectory.clear()
    else:
      # Attempt to yield a replay element
      if replay_element := self._make_replay_element():
        yield replay_element
      # If the transition truncates the trajectory then clear it
      # We don't yield n-1, n-2, ..., 1 step returns as
      # this transition is not terminal.
      if transition.episode_end:
        self._trajectory.clear()

  def clear(self) -> None:
    """Clear the accumulator."""
    self._trajectory.clear()

  def to_state_dict(self) -> dict[str, Any]:
    """Serialize to a state dictionary."""
    # We'll serialize each transition as a flat representation of its PyTree.
    # This will encode the transitions as:
    #   [[action, reward, is_terminal, episode_end], ...]
    steps = iter(self._trajectory)
    leaves, treedef = jax.tree_util.tree_flatten(next(steps, None))
    steps = [] if not leaves else [leaves, *map(treedef.flatten_up_to, steps)]
    return {'trajectory': steps, 'treedef': pickle.dumps(treedef)}

  def from_state_dict(self, state_dict: dict[str, Any]) -> None:
    """Mutate accumulator with data from `state_dict`."""
    # Load the treedef we flattened with
    treedef: jax.tree_util.PyTreeDef = pickle.loads(state_dict['treedef'])

    # We can now unflatten each transition
    # It's encoded as:
    #   [[action, reward, is_terminal, episode_end], ...]
    # After this transformation we'll have:
    #   [TransitionElement(None, action, reward, is_terminal, episode_end), ...]
    transitions = map(treedef.unflatten, state_dict['trajectory'])
    self._trajectory.clear()
    self._trajectory.extend(transitions)

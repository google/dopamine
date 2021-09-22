# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The standard DQN replay memory.
This implementation is an out-of-graph replay memory + in-graph wrapper. It
supports vanilla n-step updates of the form typically found in the literature,
i.e. where rewards are accumulated for n steps and the intermediate trajectory
is not exposed to the agent. This does not allow, for example, performing
off-policy corrections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip
import math
import os
import pickle
import time

from absl import logging
import numpy as np
import tensorflow as tf

import gin.tf
import jax
from jax import numpy as jnp
import functools

from dopamine.labs.atari_100k.replay_memory import parallel_deterministic_sum_tree as sum_tree

# Defines a type describing part of the tuple returned by the replay
# memory. Each element of the tuple is a tensor of shape [batch, ...] where
# ... is defined the 'shape' field of ReplayElement. The tensor type is
# given by the 'type' field. The 'name' field is for convenience and ease of
# debugging.
ReplayElement = (
    collections.namedtuple('shape_type', ['name', 'shape', 'type']))

# A prefix that can not collide with variable names for checkpoint files.
STORE_FILENAME_PREFIX = '$store$_'

# This constant determines how many iterations a checkpoint is kept for.
CHECKPOINT_DURATION = 4


def modulo_range(start, length, modulo):
  for i in range(length):
    yield (start + i) % modulo


def invalid_range(cursor, replay_capacity, stack_size, update_horizon):
  """Returns a array with the indices of cursor-related invalid transitions.
    There are update_horizon + stack_size invalid indices:
        - The update_horizon indices before the cursor, because we do not have a
            valid N-step transition (including the next state).
        - The stack_size indices on or immediately after the cursor.
    If N = update_horizon, K = stack_size, and the cursor is at c, invalid
    indices are:
        c - N, c - N + 1, ..., c, c + 1, ..., c + K - 1.
    It handles special cases in a circular buffer in the beginning and the end.
    Args:
        cursor: int, the position of the cursor.
        replay_capacity: int, the size of the replay memory.
        stack_size: int, the size of the stacks returned by the replay memory.
        update_horizon: int, the agent's update horizon.
    Returns:
        np.array of size stack_size with the invalid indices.
    """
  assert cursor < replay_capacity
  return np.array([(cursor - update_horizon + i) % replay_capacity
                   for i in range(stack_size + update_horizon)])


@gin.configurable
class DeterministicOutOfGraphTemporalReplayBuffer(object):
  """A simple out-of-graph Replay Buffer.
    Stores transitions, state, action, reward, next_state, terminal (and any
    extra contents specified) in a circular buffer and provides a uniform
    transition sampling function.
    When the states consist of stacks of observations storing the states is
    inefficient. This class writes observations and constructs the stacked states
    at sample time.
    Attributes:
        add_count: int, counter of how many transitions have been added (including
            the blank ones at the beginning of an episode).
        invalid_range: np.array, an array with the indices of cursor-related invalid
            transitions
    """

  def __init__(self,
               observation_shape,
               stack_size,
               replay_capacity,
               batch_size,
               jumps,
               update_horizon=1,
               gamma=0.99,
               max_sample_attempts=1000,
               extra_storage_types=None,
               observation_dtype=np.uint8,
               terminal_dtype=np.uint8,
               action_shape=(),
               action_dtype=np.int32,
               reward_shape=(),
               reward_dtype=np.float32):
    """Initializes OutOfGraphReplayBuffer.
        Args:
            observation_shape: tuple of ints.
            stack_size: int, number of frames to use in state stack.
            replay_capacity: int, number of transitions to keep in memory.
            batch_size: int.
            update_horizon: int, length of update ('n' in n-step update).
            gamma: int, the discount factor.
            max_sample_attempts: int, the maximum number of attempts allowed to
                get a sample.
            extra_storage_types: list of ReplayElements defining the type of the extra
                contents that will be stored and returned by sample_transition_batch.
            observation_dtype: np.dtype, type of the observations. Defaults to
                np.uint8 for Atari 2600.
            terminal_dtype: np.dtype, type of the terminals. Defaults to np.uint8 for
                Atari 2600.
            action_shape: tuple of ints, the shape for the action vector. Empty tuple
                means the action is a scalar.
            action_dtype: np.dtype, type of elements in the action.
            reward_shape: tuple of ints, the shape of the reward vector. Empty tuple
                means the reward is a scalar.
            reward_dtype: np.dtype, type of elements in the reward.
        Raises:
            ValueError: If replay_capacity is too small to hold at least one
                transition.
        """
    assert isinstance(observation_shape, tuple)
    if replay_capacity < update_horizon + stack_size:
      raise ValueError('There is not enough capacity to cover '
                       'update_horizon and stack_size.')

    logging.info('Creating a %s replay memory with the following parameters:',
                 self.__class__.__name__)
    logging.info('\t observation_shape: %s', str(observation_shape))
    logging.info('\t observation_dtype: %s', str(observation_dtype))
    logging.info('\t terminal_dtype: %s', str(terminal_dtype))
    logging.info('\t stack_size: %d', stack_size)
    logging.info('\t replay_capacity: %d', replay_capacity)
    logging.info('\t batch_size: %d', batch_size)
    logging.info('\t update_horizon: %d', update_horizon)
    logging.info('\t gamma: %f', gamma)

    self._action_shape = action_shape
    self._action_dtype = action_dtype
    self._reward_shape = reward_shape
    self._reward_dtype = reward_dtype
    self._observation_shape = observation_shape
    self._stack_size = stack_size
    self._state_shape = self._observation_shape + (self._stack_size,)
    self._replay_capacity = replay_capacity
    self._batch_size = batch_size
    self._update_horizon = update_horizon
    self._gamma = gamma
    self._observation_dtype = observation_dtype
    self._terminal_dtype = terminal_dtype
    self._max_sample_attempts = max_sample_attempts
    self._jumps = jumps
    if extra_storage_types:
      self._extra_storage_types = extra_storage_types
    else:
      self._extra_storage_types = []
    self._create_storage()
    self.add_count = np.array(0)
    self.invalid_range = np.zeros((self._stack_size))
    # When the horizon is > 1, we compute the sum of discounted rewards as a dot
    # product using the precomputed vector <gamma^0, gamma^1, ..., gamma^{n-1}>.
    self._cumulative_discount_vector = np.array(
        [math.pow(self._gamma, n) for n in range(update_horizon)],
        dtype=np.float32)
    self._next_experience_is_episode_start = True
    self._episode_end_indices = set()

  def _create_storage(self):
    """Creates the numpy arrays used to store transitions.
        """
    self._store = {}
    for storage_element in self.get_storage_signature():
      array_shape = [self._replay_capacity] + list(storage_element.shape)
      self._store[storage_element.name] = np.empty(
          array_shape, dtype=storage_element.type)

  def get_add_args_signature(self):
    """The signature of the add function.
        Note - Derived classes may return a different signature.
        Returns:
            list of ReplayElements defining the type of the argument signature needed
                by the add function.
        """
    return self.get_storage_signature()

  def get_storage_signature(self):
    """Returns a default list of elements to be stored in this replay memory.
        Note - Derived classes may return a different signature.
        Returns:
            list of ReplayElements defining the type of the contents stored.
        """
    storage_elements = [
        ReplayElement('observation', self._observation_shape,
                      self._observation_dtype),
        ReplayElement('action', self._action_shape, self._action_dtype),
        ReplayElement('reward', self._reward_shape, self._reward_dtype),
        ReplayElement('terminal', (), self._terminal_dtype)
    ]

    for extra_replay_element in self._extra_storage_types:
      storage_elements.append(extra_replay_element)
    return storage_elements

  def _add_zero_transition(self):
    """Adds a padding transition filled with zeros (Used in episode beginnings).
        """
    zero_transition = []
    for element_type in self.get_add_args_signature():
      zero_transition.append(
          np.zeros(element_type.shape, dtype=element_type.type))
    self._episode_end_indices.discard(self.cursor())  # If present
    self._add(*zero_transition)

  def add(self,
          observation,
          action,
          reward,
          terminal,
          *args,
          priority=None,
          episode_end=False):
    """Adds a transition to the replay memory.
        This function checks the types and handles the padding at the beginning of
        an episode. Then it calls the _add function.
        Since the next_observation in the transition will be the observation added
        next there is no need to pass it.
        If the replay memory is at capacity the oldest transition will be discarded.
        Args:
            observation: np.array with shape observation_shape.
            action: int, the action in the transition.
            reward: float, the reward received in the transition.
            terminal: np.dtype, acts as a boolean indicating whether the transition
                                was terminal (1) or not (0).
            *args: extra contents with shapes and dtypes according to
                extra_storage_types.
            priority: float, unused in the circular replay buffer, but may be used
                in child classes like PrioritizedReplayBuffer.
            episode_end: bool, whether this experience is the last experience in
                the episode. This is useful for tasks that terminate due to time-out,
                but do not end on a terminal state. Overloading 'terminal' may not
                be sufficient in this case, since 'terminal' is passed to the agent
                for training. 'episode_end' allows the replay buffer to determine
                episode boundaries without passing that information to the agent.
        """
    if priority is not None:
      args = args + (priority,)

    self._check_add_types(observation, action, reward, terminal, *args)
    if self._next_experience_is_episode_start:
      for _ in range(self._stack_size - 1):
        # Child classes can rely on the padding transitions being filled with
        # zeros. This is useful when there is a priority argument.
        self._add_zero_transition()
      self._next_experience_is_episode_start = False

    if episode_end or terminal:
      self._episode_end_indices.add(self.cursor())
      self._next_experience_is_episode_start = True
    else:
      self._episode_end_indices.discard(self.cursor())  # If present

    self._add(observation, action, reward, terminal, *args)

  def _add(self, *args):
    """Internal add method to add to the storage arrays.
        Args:
            *args: All the elements in a transition.
        """
    self._check_args_length(*args)
    transition = {
        e.name: args[idx] for idx, e in enumerate(self.get_add_args_signature())
    }
    self._add_transition(transition)

  def _add_transition(self, transition):
    """Internal add method to add transition dictionary to storage arrays.
        Args:
            transition: The dictionary of names and values of the transition
                                    to add to the storage.
        """
    cursor = self.cursor()
    for arg_name in transition:
      self._store[arg_name][cursor] = transition[arg_name]

    self.add_count += 1
    self.invalid_range = invalid_range(self.cursor(), self._replay_capacity,
                                       self._stack_size, self._update_horizon)

  def _check_args_length(self, *args):
    """Check if args passed to the add method have the same length as storage.
        Args:
            *args: Args for elements used in storage.
        Raises:
            ValueError: If args have wrong length.
        """
    if len(args) != len(self.get_add_args_signature()):
      raise ValueError('Add expects {} elements, received {}'.format(
          len(self.get_add_args_signature()), len(args)))

  def _check_add_types(self, *args):
    """Checks if args passed to the add method match those of the storage.
        Args:
            *args: Args whose types need to be validated.
        Raises:
            ValueError: If args have wrong shape or dtype.
        """
    self._check_args_length(*args)
    for i, (arg_element, store_element) in enumerate(
        zip(args, self.get_add_args_signature())):
      if isinstance(arg_element, np.ndarray):
        arg_shape = arg_element.shape
      elif isinstance(arg_element, tuple) or isinstance(arg_element, list):
        # TODO(b/80536437). This is not efficient when arg_element is a list.
        arg_shape = np.array(arg_element).shape
      else:
        # Assume it is scalar.
        arg_shape = tuple()
      store_element_shape = tuple(store_element.shape)
      if arg_shape != store_element_shape:
        raise ValueError('arg {} has shape {}, expected {}'.format(
            i, arg_shape, store_element_shape))

  def is_empty(self):
    """Is the Replay Buffer empty?"""
    return self.add_count == 0

  def is_full(self):
    """Is the Replay Buffer full?"""
    return self.add_count >= self._replay_capacity

  def cursor(self):
    """Index to the location where the next transition will be written."""
    return self.add_count % self._replay_capacity

  def parallel_get_stack(self, indices, element_name):
    array = self._store[element_name]
    result = np.take(
        array,
        np.arange(-self._stack_size + 1, 1)[:, None] + indices[None, :],
        axis=0,
        mode="wrap")
    result = np.moveaxis(result, 0, -1)
    return result

  def get_observation_stack(self, index):
    return self._get_element_stack(index, 'observation')

  def get_range(self, array, start_index, end_index):
    """Returns the range of array at the index handling wraparound if necessary.
        Args:
            array: np.array, the array to get the stack from.
            start_index: int, index to the start of the range to be returned. Range
                will wraparound if start_index is smaller than 0.
            end_index: int, exclusive end index. Range will wraparound if end_index
                exceeds replay_capacity.
        Returns:
            np.array, with shape [end_index - start_index, array.shape[1:]].
        """
    assert end_index > start_index, 'end_index must be larger than start_index'
    assert end_index >= 0
    assert start_index < self._replay_capacity
    if not self.is_full():
      assert end_index <= self.cursor(), (
          'Index {} has not been added.'.format(start_index))

    # Fast slice read when there is no wraparound.
    if start_index % self._replay_capacity < end_index % self._replay_capacity:
      return_array = array[start_index:end_index, ...]
    # Slow list read.
    else:
      indices = [(start_index + i) % self._replay_capacity
                 for i in range(end_index - start_index)]
      return_array = array[indices, ...]
    return return_array

  def get_observation_stack(self, index):
    return self._get_element_stack(index, 'observation')

  def _get_element_stack(self, index, element_name):
    state = self.get_range(self._store[element_name],
                           index - self._stack_size + 1, index + 1)
    # The stacking axis is 0 but the agent expects as the last axis.
    return np.moveaxis(state, 0, -1)

  def get_terminal_stack(self, index):
    return self.get_range(self._store['terminal'], index - self._stack_size + 1,
                          index + 1)

  def is_valid_transition(self, index):
    """Checks if the index contains a valid transition.
        Checks for collisions with the end of episodes and the current position
        of the cursor.
        Args:
            index: int, the index to the state in the transition.
        Returns:
            Is the index valid: Boolean.
        """
    index = int(index)
    # Check the index is in the valid range
    if index < 0 or index >= self._replay_capacity:
      return False
    if not self.is_full():
      # The indices and next_indices must be smaller than the cursor.
      if index >= self.cursor() - self._update_horizon - self._jumps:
        return False
      # The first few indices contain the padding states of the first episode.
      if index < self._stack_size - 1:
        return False

    # Skip transitions that straddle the cursor.
    if index in set(self.invalid_range):
      return False

    # If there are terminal flags in any other frame other than the last one
    # the stack is not valid, so don't sample it.
    if self.get_terminal_stack(index)[:-1].any():
      return False

    # If the episode ends before the update horizon, without a terminal signal,
    # it is invalid.
    for i in modulo_range(index, self._update_horizon, self._replay_capacity):
      if i in self._episode_end_indices and not self._store['terminal'][i]:
        return False

    return True

  def _create_batch_arrays(self, batch_size):
    """Create a tuple of arrays with the type of get_transition_elements.
        When using the WrappedReplayBuffer with staging enabled it is important to
        create new arrays every sample because StaginArea keeps a pointer to the
        returned arrays.
        Args:
            batch_size: (int) number of transitions returned. If None the default
                batch_size will be used.
        Returns:
            Tuple of np.arrays with the shape and type of get_transition_elements.
        """
    transition_elements = self.get_transition_elements(batch_size)
    batch_arrays = []
    for element in transition_elements:
      batch_arrays.append(np.empty(element.shape, dtype=element.type))
    return tuple(batch_arrays)

  def sample_index_batch(self, batch_size):
    """Returns a batch of valid indices sampled uniformly.

        Args:
            batch_size: int, number of indices returned.

        Returns:
            list of ints, a batch of valid indices sampled uniformly.

        Raises:
            RuntimeError: If the batch was not constructed after maximum number of
                tries.
        """
    self._rng, rng = jax.random.split(self._rng)
    if self.is_full():
      # add_count >= self._replay_capacity > self._stack_size
      min_id = self.cursor() - self._replay_capacity + self._stack_size - 1
      max_id = self.cursor() - self._update_horizon - self._jumps
    else:
      # add_count < self._replay_capacity
      min_id = self._stack_size - 1
      max_id = self.cursor() - self._update_horizon - self._jumps
      if max_id <= min_id:
        raise RuntimeError('Cannot sample a batch with fewer than stack size '
                           '({}) + update_horizon ({}) transitions.'.format(
                               self._stack_size, self._update_horizon))
    indices = jax.random.randint(rng, (batch_size,), min_id,
                                 max_id) % self._replay_capacity
    allowed_attempts = self._max_sample_attempts
    indices = np.array(indices)
    for i in range(len(indices)):
      if not self.is_valid_transition(indices[i]):
        if allowed_attempts == 0:
          raise RuntimeError(
              'Max sample attempts: Tried {} times but only sampled {}'
              ' valid indices. Batch size is {}'.format(
                  self._max_sample_attempts, i, batch_size))
        index = indices[i]
        while not self.is_valid_transition(index) and allowed_attempts > 0:
          # If index i is not valid keep sampling others. Note that this
          # is not stratified.
          self._rng, rng = jax.random.split(self._rng)
          index = jax.random.randint(rng,
                                     (), min_id, max_id) % self._replay_capacity
          allowed_attempts -= 1
        indices[i] = index
    return indices

  def restore_leading_dims(self, batch_size, jumps, tensor):
    return tensor.reshape(batch_size, jumps, *tensor.shape[1:])

  def sample_transition_batch(self,
                              rng,
                              batch_size=None,
                              indices=None,
                              jumps=None):
    """Returns a batch of transitions (including any extra contents).
        If get_transition_elements has been overridden and defines elements not
        stored in self._store, an empty array will be returned and it will be
        left to the child class to fill it. For example, for the child class
        OutOfGraphPrioritizedReplayBuffer, the contents of the
        sampling_probabilities are stored separately in a sum tree.
        When the transition is terminal next_state_batch has undefined contents.
        NOTE: This transition contains the indices of the sampled elements. These
        are only valid during the call to sample_transition_batch, i.e. they may
        be used by subclasses of this replay buffer but may point to different data
        as soon as sampling is done.
        Args:
            batch_size: int, number of transitions returned. If None, the default
                batch_size will be used.
            indices: None or list of ints, the indices of every transition in the
                batch. If None, sample the indices uniformly.
        Returns:
            transition_batch: tuple of np.arrays with the shape and type as in
                get_transition_elements().
        Raises:
            ValueError: If an element to be sampled is missing from the replay buffer.
        """
    self._rng = rng
    if batch_size is None:
      batch_size = self._batch_size
    if jumps is None:
      jumps = self._jumps
    if indices is None:
      indices = self.sample_index_batch(batch_size)
    assert len(indices) == batch_size
    transition_elements = self.get_transition_elements(batch_size)
    state_indices = indices[:, None] + np.arange(jumps)[None, :]
    state_indices = state_indices.reshape(batch_size * jumps)

    # shape: horizon X batch_size*jumps
    # Offset by one; a `d
    trajectory_indices = (np.arange(-1, self._update_horizon - 1)[:, None] +
                          state_indices[None, :]) % self._replay_capacity
    trajectory_terminals = self._store["terminal"][trajectory_indices]
    trajectory_terminals[0, :] = 0
    is_terminal_transition = trajectory_terminals.any(0)
    valid_mask = (1 - trajectory_terminals).cumprod(0)
    trajectory_discount_vector = valid_mask * self._cumulative_discount_vector[:,
                                                                               None]
    trajectory_rewards = self._store['reward'][(trajectory_indices + 1) %
                                               self._replay_capacity]
    returns = np.sum(trajectory_discount_vector * trajectory_rewards, axis=0)

    next_indices = (state_indices +
                    self._update_horizon) % self._replay_capacity
    outputs = []

    for element in transition_elements:
      name = element.name
      if name == 'state':
        output = self.parallel_get_stack(state_indices, "observation")
        output = self.restore_leading_dims(batch_size, jumps, output)
      elif name == 'reward':
        # compute the discounted sum of rewards in the trajectory.
        output = returns
        output = self.restore_leading_dims(batch_size, jumps, output)
      elif name == 'next_state':
        output = self.parallel_get_stack(next_indices, "observation")
        output = self.restore_leading_dims(batch_size, jumps, output)
      elif name == "same_trajectory":
        output = self._store["terminal"][state_indices]
        output = self.restore_leading_dims(batch_size, jumps, output)
        output[0, :] = 0
        output = (1 - output).cumprod(1)
      elif name == 'valid':
        output = np.array([self.is_valid_transition(i) for i in state_indices])
        output = self.restore_leading_dims(batch_size, jumps, output)
      elif name in ('next_action', 'next_reward'):
        output = self._store[name.lstrip('next_')][next_indices]
        output = self.restore_leading_dims(batch_size, jumps, output)
      elif element.name == 'terminal':
        output = is_terminal_transition
        output = self.restore_leading_dims(batch_size, jumps, output)
      elif name == 'indices':
        output = indices
      elif name in self._store.keys():
        output = self._store[name][state_indices]
        output = self.restore_leading_dims(batch_size, jumps, output)
      else:
        continue
      outputs.append(output)
    return outputs

  def get_transition_elements(self, batch_size=None, jumps=None):
    """Returns a 'type signature' for sample_transition_batch.
        Args:
            batch_size: int, number of transitions returned. If None, the default
                batch_size will be used.
        Returns:
            signature: A namedtuple describing the method's return type signature.
        """
    jumps = self._jumps if jumps is None else jumps
    batch_size = self._batch_size if batch_size is None else batch_size

    transition_elements = [
        ReplayElement('state', (batch_size, jumps) + self._state_shape,
                      self._observation_dtype),
        ReplayElement('action', (batch_size, jumps) + self._action_shape,
                      self._action_dtype),
        ReplayElement('reward', (batch_size, jumps) + self._reward_shape,
                      self._reward_dtype),
        ReplayElement('next_state', (batch_size, jumps) + self._state_shape,
                      self._observation_dtype),
        ReplayElement('next_action', (batch_size, jumps) + self._action_shape,
                      self._action_dtype),
        ReplayElement('next_reward', (batch_size, jumps) + self._reward_shape,
                      self._reward_dtype),
        ReplayElement('terminal', (batch_size, jumps), self._terminal_dtype),
        ReplayElement('same_trajectory', (batch_size, jumps),
                      self._terminal_dtype),
        ReplayElement('valid', (batch_size, jumps), self._terminal_dtype),
        ReplayElement('indices', (batch_size,), np.int32)
    ]
    for element in self._extra_storage_types:
      transition_elements.append(
          ReplayElement(element.name,
                        (batch_size, jumps) + tuple(element.shape),
                        element.type))
    return transition_elements

  def _generate_filename(self, checkpoint_dir, name, suffix):
    return os.path.join(checkpoint_dir, '{}_ckpt.{}.gz'.format(name, suffix))

  def _return_checkpointable_elements(self):
    """Return the dict of elements of the class for checkpointing.
        Returns:
            checkpointable_elements: dict containing all non private (starting with
            _) members + all the arrays inside self._store.
        """
    checkpointable_elements = {}
    for member_name, member in self.__dict__.items():
      if member_name == '_store':
        for array_name, array in self._store.items():
          checkpointable_elements[STORE_FILENAME_PREFIX + array_name] = array
      elif not member_name.startswith('_'):
        checkpointable_elements[member_name] = member
    return checkpointable_elements

  def save(self, checkpoint_dir, iteration_number):
    """Save the OutOfGraphReplayBuffer attributes into a file.
        This method will save all the replay buffer's state in a single file.
        Args:
            checkpoint_dir: str, the directory where numpy checkpoint files should be
                saved.
            iteration_number: int, iteration_number to use as a suffix in naming
                numpy checkpoint files.
        """
    if not tf.io.gfile.exists(checkpoint_dir):
      return

    checkpointable_elements = self._return_checkpointable_elements()

    for attr in checkpointable_elements:
      filename = self._generate_filename(checkpoint_dir, attr, iteration_number)
      with tf.io.gfile.GFile(filename, 'wb') as f:
        with gzip.GzipFile(fileobj=f) as outfile:
          # Checkpoint the np arrays in self._store with np.save instead of
          # pickling the dictionary is critical for file size and performance.
          # STORE_FILENAME_PREFIX indicates that the variable is contained in
          # self._store.
          if attr.startswith(STORE_FILENAME_PREFIX):
            array_name = attr[len(STORE_FILENAME_PREFIX):]
            np.save(outfile, self._store[array_name], allow_pickle=False)
          # Some numpy arrays might not be part of storage
          elif isinstance(self.__dict__[attr], np.ndarray):
            np.save(outfile, self.__dict__[attr], allow_pickle=False)
          else:
            pickle.dump(self.__dict__[attr], outfile)

      # After writing a checkpoint file, we garbage collect the checkpoint file
      # that is four versions old.
      stale_iteration_number = iteration_number - CHECKPOINT_DURATION
      if stale_iteration_number >= 0:
        stale_filename = self._generate_filename(checkpoint_dir, attr,
                                                 stale_iteration_number)
        try:
          tf.io.gfile.remove(stale_filename)
        except tf.errors.NotFoundError:
          pass

  def load(self, checkpoint_dir, suffix):
    """Restores the object from bundle_dictionary and numpy checkpoints.
        Args:
            checkpoint_dir: str, the directory where to read the numpy checkpointed
                files from.
            suffix: str, the suffix to use in numpy checkpoint files.
        Raises:
            NotFoundError: If not all expected files are found in directory.
        """
    save_elements = self._return_checkpointable_elements()
    # We will first make sure we have all the necessary files available to avoid
    # loading a partially-specified (i.e. corrupted) replay buffer.
    for attr in save_elements:
      filename = self._generate_filename(checkpoint_dir, attr, suffix)
      if not tf.io.gfile.exists(filename):
        raise tf.errors.NotFoundError(None, None,
                                      'Missing file: {}'.format(filename))
    # If we've reached this point then we have verified that all expected files
    # are available.
    for attr in save_elements:
      filename = self._generate_filename(checkpoint_dir, attr, suffix)
      with tf.io.gfile.GFile(filename, 'rb') as f:
        with gzip.GzipFile(fileobj=f) as infile:
          if attr.startswith(STORE_FILENAME_PREFIX):
            array_name = attr[len(STORE_FILENAME_PREFIX):]
            self._store[array_name] = np.load(infile, allow_pickle=False)
          elif isinstance(self.__dict__[attr], np.ndarray):
            self.__dict__[attr] = np.load(infile, allow_pickle=False)
          else:
            self.__dict__[attr] = pickle.load(infile)


@gin.configurable
class DeterministicOutOfGraphPrioritizedTemporalReplayBuffer(
    DeterministicOutOfGraphTemporalReplayBuffer):
  """Deterministic version of prioritized replay buffer."""

  def __init__(self,
               observation_shape,
               stack_size,
               replay_capacity,
               batch_size,
               update_horizon=1,
               jumps=0,
               gamma=0.99,
               max_sample_attempts=1000,
               extra_storage_types=None,
               observation_dtype=np.uint8,
               terminal_dtype=np.uint8,
               action_shape=(),
               action_dtype=np.int32,
               reward_shape=(),
               reward_dtype=np.float32):
    super().__init__(
        observation_shape=observation_shape,
        stack_size=stack_size,
        replay_capacity=replay_capacity,
        batch_size=batch_size,
        update_horizon=update_horizon,
        gamma=gamma,
        max_sample_attempts=max_sample_attempts,
        extra_storage_types=extra_storage_types,
        observation_dtype=observation_dtype,
        terminal_dtype=terminal_dtype,
        jumps=jumps,
        action_shape=action_shape,
        action_dtype=action_dtype,
        reward_shape=reward_shape,
        reward_dtype=reward_dtype)

    self.sum_tree = sum_tree.DeterministicSumTree(replay_capacity)

  def get_add_args_signature(self):
    """The signature of the add function."""
    parent_add_signature = super().get_add_args_signature()
    add_signature = parent_add_signature + [
        ReplayElement('priority', (), np.float32)
    ]
    return add_signature

  def _add(self, *args):
    """Internal add method to add to the underlying memory arrays."""
    self._check_args_length(*args)

    # Use Schaul et al.'s (2015) scheme of setting the priority of new elements
    # to the maximum priority so far.
    # Picks out 'priority' from arguments and adds it to the sum_tree.
    transition = {}
    for i, element in enumerate(self.get_add_args_signature()):
      if element.name == 'priority':
        priority = args[i]
      else:
        transition[element.name] = args[i]

    self.sum_tree.set(self.cursor(), priority)
    super()._add_transition(transition)

  def sample_index_batch(self, batch_size):
    """Returns a batch of valid indices sampled as in Schaul et al. (2015)."""
    # Sample stratified indices. Some of them might be invalid.
    indices = self.sum_tree.stratified_sample(batch_size, self._rng)
    allowed_attempts = self._max_sample_attempts
    indices = np.array(indices)
    for i in range(len(indices)):
      if not self.is_valid_transition(indices[i]):
        if allowed_attempts == 0:
          raise RuntimeError(
              'Max sample attempts: Tried {} times but only sampled {}'
              ' valid indices. Batch size is {}'.format(
                  self._max_sample_attempts, i, batch_size))
        index = indices[i]
        while not self.is_valid_transition(index) and allowed_attempts > 0:
          # If index i is not valid keep sampling others.
          # Note that this is not stratified.
          self._rng, rng = jax.random.split(self._rng)
          index = int(self.sum_tree.sample(rng=rng))
          allowed_attempts -= 1
        indices[i] = index
    return indices

  def sample_transition_batch(self, rng, batch_size=None, indices=None):
    """Returns a batch of transitions with extra storage and the priorities."""
    transition = super().sample_transition_batch(rng, batch_size, indices)

    # By convention, the indices are always the last element of the batch
    indices = transition[-1]
    priority = self.get_priority(indices)
    transition.append(priority)
    return transition

  def set_priority(self, indices, priorities):
    """Sets the priority of the given elements according to Schaul et al."""
    assert indices.dtype == np.int32, ('Indices must be integers, '
                                       'given: {}'.format(indices.dtype))
    for index, priority in zip(indices, priorities):
      self.sum_tree.set(index, priority)

  def get_priority(self, indices):
    """Fetches the priorities correspond to a batch of memory indices."""
    assert indices.shape, 'Indices must be an array.'
    assert indices.dtype == np.int32, ('Indices must be int32s, '
                                       'given: {}'.format(indices.dtype))
    batch_size = len(indices)
    priority_batch = np.empty((batch_size), dtype=np.float32)
    for i, memory_index in enumerate(indices):
      priority_batch[i] = self.sum_tree.get(memory_index)
    return priority_batch

  def get_transition_elements(self, batch_size=None):
    """Returns a 'type signature' for sample_transition_batch."""
    parent_transition_type = (super().get_transition_elements(batch_size))
    probablilities_type = [
        ReplayElement('sampling_probabilities', (batch_size,), np.float32)
    ]
    return parent_transition_type + probablilities_type

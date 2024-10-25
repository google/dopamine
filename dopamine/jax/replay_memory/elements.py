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
"""Various containers used by the replay buffer code."""

import typing
from typing import Any, NewType, Optional, Protocol

from flax import struct
import numpy as np
import numpy.typing as npt
import snappy

ReplayItemID = NewType('ReplayItemID', int)


class TransitionElement(typing.NamedTuple):
  observation: Optional[npt.NDArray[Any]]
  action: int
  reward: float
  is_terminal: bool
  episode_end: bool = False


@typing.runtime_checkable
class ReplayElementProtocol(Protocol):

  def pack(self) -> 'ReplayElementProtocol':
    ...

  def unpack(self) -> 'ReplayElementProtocol':
    ...

  @property
  def is_compressed(self) -> bool:
    ...


def compress(buffer: npt.NDArray) -> npt.NDArray:
  """Compress a numpy array using snappy.

  Args:
    buffer: npt.NDArray, buffer to compress.

  Returns:
    Numpy structured array consisting of the following fields:
      data: compressed data in bytes
      shape: the shape of the uncompressed array
      dtype: a string representation of the dtype
  """
  # Compress the numpy array using snappy
  if not buffer.flags['C_CONTIGUOUS']:
    buffer = buffer.copy(order='C')
  compressed = np.frombuffer(snappy.compress(buffer), dtype=np.uint8)
  # Construct the numpy structured array
  return np.array(
      (compressed, buffer.shape, buffer.dtype.str),
      dtype=[
          ('data', 'u1', compressed.shape),
          ('shape', 'i4', (len(buffer.shape),)),
          ('dtype', f'S{len(buffer.dtype.str)}'),
      ],
  )


def uncompress(compressed: npt.NDArray) -> npt.NDArray:
  """Uncompress a numpy array that has been compressed via `compress`.

  Args:
    compressed: npt.NDArray, numpy structured array with data, shape, and dtype.

  Returns:
    Uncompressed npt.NDArray
  """
  # Create shape tuple
  shape = tuple(compressed['shape'])
  # Get the dtype string
  dtype = compressed['dtype'].item()
  # Get the compressed bytes and uncompress them using snappy
  compressed_bytes = compressed['data'].tobytes()
  uncompressed = snappy.uncompress(compressed_bytes)
  # Construct the numpy array
  return np.ndarray(shape=shape, dtype=dtype, buffer=uncompressed)


class ReplayElement(ReplayElementProtocol, struct.PyTreeNode):
  """A single replay transition element supporting compression."""

  state: npt.NDArray[np.float64]
  action: npt.NDArray[np.int_] | npt.NDArray[np.float64] | int
  reward: npt.NDArray[np.float64] | float
  next_state: npt.NDArray[np.float64]
  is_terminal: npt.NDArray[np.bool_] | bool
  episode_end: npt.NDArray[np.bool_] | bool

  def pack(self) -> 'ReplayElement':
    # NOTE: pytype has a problem subclassing generics.
    # This should be solved in Py311 with `typing.Self`
    # pytype: disable=attribute-error
    return self.replace(
        state=compress(self.state),
        next_state=compress(self.next_state),
    )
    # pytype: enable=attribute-error

  def unpack(self) -> 'ReplayElement':
    # NOTE: pytype has a problem subclassing generics.
    # This should be solved in Py311 with `typing.Self`
    # pytype: disable=attribute-error
    return self.replace(
        state=uncompress(self.state),
        next_state=uncompress(self.next_state),
    )
    # pytype: enable=attribute-error

  @property
  def is_compressed(self) -> bool:
    # NOTE: pytype has a problem subclassing generics.
    # This should be solved in Py311 with `typing.Self`
    # pytype: disable=attribute-error
    # As per the numpy documentation the recommended way to check for a
    # structured array is to check if `dtype.names is not None`.
    # See: https://numpy.org/doc/stable/user/basics.rec.html
    return (
        self.state.dtype.names is not None
        and self.next_state.dtype.names is not None
    )
    # pytype: enable=attribute-error

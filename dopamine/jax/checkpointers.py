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
"""Checkpointer using Orbax and MessagePack.

Usage:
  1) Implement `to_state_dict` and `from_state_dict` on your class.
  2) Create an `orbax.CheckpointManager` and use the `CheckpointHandler`.

  To save: call `save` on your checkpoint manager passing the class instance.
  To restore: call `restore` on your checkpoint manager passing the class
              instance.

Example:

  ```py
  from typing import Dict, Any

  import numpy.typing as npt
  from orbax import checkpoint as orbax
  from dopamine.jax import checkpoint

  class MyClass:
    def __init__(self, data: npt.NDArray):
      self._data = data

    def to_state_dict(self) -> Dict[str, Any]:
      return {'data': self._data}

    def from_state_dict(self, state_dict: Dict[str, Any]) -> None:
      self._data = state_dict['data']

  checkpoint_manager = orbax.CheckpointManager(
      my_workdir,
      checkpointers=checkpoint.Checkpointer(),
  )

  instance = MyClass(np.array([1, 2, 3]))
  checkpoint_manager.save(1, instance)
  # Creates a copy of instance
  restored_instance = checkpoint_manager.restore(1, instance)

  # If you don't want to create a copy:
  instance.from_state_dict(checkpoint_manager.restore(1))
  ```
"""

import copy
import functools
import typing
from typing import Any, Dict, Generic, Optional, Protocol, TypeVar, Union

from dopamine.jax import serialization
from etils import epath
import msgpack
from orbax import checkpoint


@typing.runtime_checkable
class Checkpointable(Protocol):
  """Checkpointable protocol. Must implement to_state_dict, from_state_dict."""

  def to_state_dict(self) -> Dict[str, Any]:
    ...

  def from_state_dict(self, state_dict: Dict[str, Any]) -> None:
    ...


CheckpointableT = TypeVar('CheckpointableT', bound=Checkpointable)


class CheckpointHandler(checkpoint.CheckpointHandler, Generic[CheckpointableT]):
  """Checkpointable protocol checkpoint handler."""

  def __init__(self, filename: str = 'checkpoint.msgpack') -> None:
    self._filename = filename

  def save(self, directory: epath.Path, item: CheckpointableT) -> None:
    if not isinstance(item, Checkpointable):
      raise NotImplementedError(f'Item {item!r} must implement Checkpointable')
    directory.mkdir(exist_ok=True, parents=True)
    filename = directory / self._filename

    # Get bytes using MsgPack
    packed = msgpack.packb(
        item.to_state_dict(),
        default=serialization.encode,
        strict_types=False,
        use_bin_type=True,
    )
    filename.write_bytes(packed)

  @typing.overload
  def restore(
      self, directory: epath.Path, item: CheckpointableT
  ) -> CheckpointableT:
    ...

  @typing.overload
  def restore(self, directory: epath.Path, item: None = None) -> Dict[str, Any]:
    ...

  def restore(
      self, directory: epath.Path, item: Optional[CheckpointableT] = None
  ) -> Union[CheckpointableT, Dict[str, Any]]:
    filename = directory / self._filename
    state_dict = msgpack.unpackb(
        filename.read_bytes(),
        object_hook=serialization.decode,
        raw=False,
        strict_map_key=False,
    )

    if item is None:
      return state_dict

    item = copy.deepcopy(item)
    item.from_state_dict(state_dict)
    return item

  def structure(self, directory: epath.Path) -> None:
    return None


# pylint: disable=g-long-lambda
# Orbax requires a `Checkpointer` object. This wraps a `CheckpointHandler`.
# To make it easier for end-users we'll supply a `Checkpointer` that already
# performs this instantiation.
Checkpointer = functools.partial(checkpoint.Checkpointer, CheckpointHandler())
# pylint: enable=g-long-lambda

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
"""Tests for replay element data structure."""

from absl.testing import absltest
from absl.testing import parameterized
from dopamine.jax.replay_memory import elements
import numpy as np


mock = absltest.mock


class ElementsTest(parameterized.TestCase):

  def test_pack_unpack(self) -> None:
    """Simple test case that packs and unpacks a replay element."""
    state = np.zeros((84, 84, 4), dtype=np.uint8)
    next_state = np.ones((84, 84, 4), dtype=np.uint8)
    action = 1
    reward = 1.0
    episode_end = False

    element = elements.ReplayElement(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        is_terminal=episode_end,
        episode_end=episode_end,
    )

    packed = element.pack()
    assert packed.is_compressed
    assert packed.action == action
    assert packed.reward == reward
    assert packed.is_terminal == packed.episode_end == episode_end

    unpacked = packed.unpack()
    assert not unpacked.is_compressed
    assert unpacked.action == action
    assert unpacked.reward == reward
    assert unpacked.is_terminal == unpacked.episode_end == episode_end

    np.testing.assert_array_equal(unpacked.state, state)
    np.testing.assert_array_equal(unpacked.next_state, next_state)


if __name__ == '__main__':
  absltest.main()

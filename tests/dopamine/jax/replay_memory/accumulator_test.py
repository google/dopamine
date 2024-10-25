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
"""Tests for accumulator.py."""

from absl.testing import absltest
from absl.testing import parameterized
from dopamine.jax.replay_memory import accumulator
from dopamine.jax.replay_memory import elements
import numpy as np

mock = absltest.mock


class TransitionAccumulatorTest(parameterized.TestCase):

  def _verify_accumulator_transition(
      self,
      accumulator_transition: elements.TransitionElement,
      transition: elements.TransitionElement | None = None,
  ):
    transition = self._sample_transition if transition is None else transition
    # The observation is removed from elements_accumulator since it is stored in
    # the stack.
    np.testing.assert_array_equal(
        accumulator_transition.observation, transition.observation
    )
    self.assertEqual(accumulator_transition.action, transition.action)
    self.assertEqual(accumulator_transition.reward, transition.reward)
    self.assertEqual(
        accumulator_transition.is_terminal, int(transition.is_terminal)
    )
    self.assertEqual(
        accumulator_transition.episode_end, int(transition.episode_end)
    )

  def _add(
      self,
      acc: accumulator.TransitionAccumulator,
      transition: elements.TransitionElement,
  ):
    return list(acc.accumulate(transition))

  def setUp(self):
    super().setUp()
    self._stack_size = 4
    self._update_horizon = 3
    self._gamma = 0.9
    self._obs = np.ones((4, 3))
    self._sample_transition = elements.TransitionElement(
        self._obs, 1, 3.14, False, False
    )

  def testInitializer(self):
    acc = accumulator.TransitionAccumulator(
        self._stack_size, self._update_horizon, self._gamma
    )
    self.assertEmpty(acc._trajectory)

  def testReset(self):
    acc = accumulator.TransitionAccumulator(
        self._stack_size, self._update_horizon, self._gamma
    )
    # We add enough transitions for a valid trajectory.
    for i in range(self._update_horizon + self._stack_size + 1):
      self._add(
          acc, elements.TransitionElement(self._obs * i, i, i, False, False)
      )
    self.assertLen(acc._trajectory, self._stack_size + self._update_horizon)
    self._add(acc, elements.TransitionElement(self._obs * 8, 8, 8, True, True))
    self.assertEmpty(acc._trajectory)

  def testOneElementTrajectoriesAreInvalid(self):
    acc = accumulator.TransitionAccumulator(
        self._stack_size, self._update_horizon, self._gamma
    )
    timesteps = self._add(acc, self._sample_transition)
    self.assertEmpty(timesteps)
    self._verify_accumulator_transition(acc._trajectory[0])
    # The trajectory accumulator shouldn't have been cleared
    self.assertNotEmpty(acc._trajectory)

  @parameterized.named_parameters(
      dict(testcase_name='OneStep', n=1, valid=True),
      dict(testcase_name='MultiStep', n=3, valid=False),
      dict(testcase_name='MultiStepGreaterThanStack', n=5, valid=False),
  )
  def testAddSingleTrajectory(self, n: int, valid: bool):
    acc = accumulator.TransitionAccumulator(self._stack_size, n, self._gamma)
    # We need at least two transitions per trajectory, otherwise there is no
    # next state.
    self.assertListEqual(self._add(acc, self._sample_transition), [])
    timesteps = self._add(acc, self._sample_transition)
    # Verify the stack now has elements.
    self.assertLen(acc._trajectory, 2)

    self._verify_accumulator_transition(acc._trajectory[0])
    self._verify_accumulator_transition(acc._trajectory[0])

    if not valid:
      self.assertEmpty(timesteps)
    else:
      self.assertLen(timesteps, 1)
      self.assertIsNotNone(timesteps[0])
      # Two last elements of the stack should be the observations just added.
      np.testing.assert_array_equal(timesteps[0].state[..., -1], self._obs)
      np.testing.assert_allclose(timesteps[0].state[..., :-1], 0)

      np.testing.assert_array_equal(timesteps[0].next_state[..., -1], self._obs)
      np.testing.assert_array_equal(timesteps[0].next_state[..., -2], self._obs)
      np.testing.assert_allclose(timesteps[0].next_state[..., :-2], 0)

      self.assertEqual(timesteps[0].action, acc._trajectory[0].action)
      self.assertEqual(timesteps[0].is_terminal, acc._trajectory[1].is_terminal)
      self.assertEqual(timesteps[0].episode_end, acc._trajectory[1].episode_end)

      self.assertEqual(
          timesteps[0].reward,
          np.sum(
              self._sample_transition.reward * (self._gamma ** np.arange(n))
          ),
      )

  @parameterized.parameters(4, 5, 6)
  def testAccumulate(self, n):
    acc = accumulator.TransitionAccumulator(self._stack_size, n, self._gamma)
    # We add enough transitions for a valid trajectory.
    for i in range(8):
      element = list(
          acc.accumulate(
              elements.TransitionElement(self._obs * i, i, i, False, False)
          )
      )
      if i > n - 1:
        self.assertLen(element, 1)
        self.assertIsNotNone(element[0])
      else:
        self.assertEmpty(element)

  def testAccumulateWithInvalidFirstTrajectory(self):
    acc = accumulator.TransitionAccumulator(self._stack_size, 1, self._gamma)
    # We add a single terminal trajectory which is invalid, and should thus
    # return None.
    element = list(
        acc.accumulate(elements.TransitionElement(self._obs, 1, 1, True, False))
    )
    self.assertEmpty(element)
    # Adding a new transition should still return None, as it is not yet valid.
    # This will also have the hidden effect of removing the first (invalid)
    # trajectory.
    element = list(
        acc.accumulate(
            elements.TransitionElement(self._obs * 2, 2, 2, False, False)
        )
    )
    self.assertEmpty(element)
    # Now adding a third transition (the second of the second trajectory) will
    # result in a valid element returned. This element will be the second
    # transition added (or the first in the second trajectory).
    element = list(
        acc.accumulate(
            elements.TransitionElement(self._obs * 3, 3, 3, False, False)
        )
    )
    self.assertLen(element, 1)
    self.assertIsNotNone(element[0])
    # The state returned will be a stack with the first observation at
    # the end.
    state = [np.zeros_like(self._obs) for _ in range(self._stack_size)]
    state[-1] = self._obs * 2
    state = np.moveaxis(state, 0, -1)
    # The next_state returned will be a stack with the last two observations at
    # the end.
    next_state = [np.zeros_like(self._obs) for _ in range(self._stack_size)]
    next_state[-2] = self._obs * 2
    next_state[-1] = self._obs * 3
    next_state = np.moveaxis(next_state, 0, -1)
    np.testing.assert_array_equal(state, element[0].state)
    np.testing.assert_array_equal(next_state, element[0].next_state)
    self.assertEqual(2, element[0].action)
    self.assertEqual(2.0, element[0].reward)
    self.assertFalse(element[0].is_terminal)
    self.assertFalse(element[0].episode_end)

  def testClear(self):
    acc = accumulator.TransitionAccumulator(
        self._stack_size, self._update_horizon, self._gamma
    )
    self._add(acc, self._sample_transition)
    self.assertNotEmpty(acc._trajectory)
    acc.clear()
    self.assertEmpty(acc._trajectory)


if __name__ == '__main__':
  absltest.main()

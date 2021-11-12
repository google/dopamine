# coding=utf-8
# Copyright 2021 The Dopamine Authors.
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
"""Tests for dopamine.labs.sac_from_pixels.deepmind_control_lib."""

import collections
from typing import Callable, Mapping
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import suite
from dm_control.rl import control
import dm_env
from dm_env import specs
from dopamine.labs.sac_from_pixels import deepmind_control_lib
import numpy as np

_MID_TIMESTEP = dm_env.TimeStep(
    step_type=dm_env.StepType.MID,
    reward=0.0,
    discount=None,
    observation={'observation': np.zeros((4,), dtype=np.float32)})
LAST_TIMESTEP = dm_env.TimeStep(
    step_type=dm_env.StepType.LAST,
    reward=-1.0,
    discount=0.0,  # Indicates a terminal state.
    observation={'observation': np.zeros((4,), dtype=np.float32)})


def get_mock_render(
    img_dtype: np.dtype = np.float32,
    img_fill_value: float = 0.0) -> Callable[[int, int, int], np.ndarray]:

  def mock_render(height: int, width: int, camera_id: int = 0):
    del camera_id  # Unused
    return np.full((height, width, 3), img_fill_value, dtype=img_dtype)

  return mock_render


# TODO(joshgreaves): Remove mock environment as I update old tests.
class MockDeepmindControlSuiteEnvironment(control.Environment):

  def __init__(self,
               img_dtype: np.dtype = np.float32,
               img_fill_value: float = 0.0,
               steps_to_terminal: int = 1):
    self._physics = mock.MagicMock()
    self._physics.render = get_mock_render(img_dtype, img_fill_value)
    self._steps_remaining = steps_to_terminal

  def action_spec(self) -> specs.BoundedArray:
    return specs.BoundedArray((4,), np.float32,
                              np.full((4,), -1.0, dtype=np.float32),
                              np.full((4,), 1.0, dtype=np.float32))

  def observation_spec(self) -> Mapping[str, specs.BoundedArray]:
    result = collections.OrderedDict()
    result['obs1'] = specs.BoundedArray((2,), np.float32,
                                        np.asarray([-1.0, -1.0]),
                                        np.asarray([1.0, 1.0]))
    result['obs2'] = specs.BoundedArray((3,), np.float32,
                                        np.asarray([-5.0, -5.0, -float('inf')]),
                                        np.asarray([5.0, 5.0,
                                                    float('inf')]))
    return result

  def reset(self) -> dm_env.TimeStep:
    observation = collections.OrderedDict()
    observation['obs1'] = np.full((2,), self._steps_remaining, dtype=np.float32)
    observation['obs2'] = np.full((3,), self._steps_remaining, dtype=np.float32)
    return dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=None,
        discount=None,
        observation=observation)

  def step(self, action: np.ndarray) -> dm_env.TimeStep:
    self._steps_remaining -= 1

    observation = collections.OrderedDict()
    observation['obs1'] = np.full((2,), self._steps_remaining, dtype=np.float32)
    observation['obs2'] = np.full((3,), self._steps_remaining, dtype=np.float32)

    if self._steps_remaining < 1:
      return dm_env.TimeStep(
          step_type=dm_env.StepType.LAST,
          reward=-1.0,
          discount=0.0,  # Indicates a terminal state.
          observation=observation)
    else:
      return dm_env.TimeStep(
          step_type=dm_env.StepType.MID,
          reward=0.0,
          discount=None,
          observation=observation)


class DeepmindControlLibTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0)

    self.env_name = 'cartpole'
    self.task_name = 'swingup'

    # The default timeout for cartpole swingup. 10 seconds at 0.01 s/step.
    self.max_steps = 1000

  def create_env(self) -> control.Environment:
    return suite.load(self.env_name, self.task_name,
                      environment_kwargs={'flat_observation': True})

  # TODO(joshgreaves): Break up test to test individual behaviors.
  def test_preprocessing(self):
    env = MockDeepmindControlSuiteEnvironment()
    env = deepmind_control_lib.DeepmindControlPreprocessing(env)

    # Test action space conversion
    np.testing.assert_equal(env.action_space.low,
                            np.full((4,), -1.0, dtype=np.float32))
    np.testing.assert_equal(env.action_space.high,
                            np.full((4,), 1.0, dtype=np.float32))

    # Test observation space conversion
    np.testing.assert_equal(
        env.observation_space.low,
        np.asarray([-1.0, -1.0, -5.0, -5.0, -np.inf], dtype=np.float32))
    np.testing.assert_equal(
        env.observation_space.high,
        np.asarray([1.0, 1.0, 5.0, 5.0, np.inf], dtype=np.float32))

    # Test reset and step
    observation = env.reset()
    np.testing.assert_equal(observation, np.ones((5,), dtype=np.float32))
    observation, reward, done, info = env.step(env.action_space.sample())
    np.testing.assert_equal(observation, np.zeros((5,), dtype=np.float32))
    self.assertEqual(reward, -1.0)
    self.assertEqual(done, True)
    self.assertEqual(info, {})
    self.assertEqual(env.game_over, True)

  def test_image_wrapper_returns_image_observation_on_reset(self):
    env = deepmind_control_lib.create_deepmind_control_environment(
        'cartpole', 'swingup', use_image_observations=True)

    state = env.reset()

    self.assertIsInstance(state, np.ndarray)
    self.assertEqual(state.shape, (84, 84, 3))
    self.assertEqual(state.dtype, np.uint8)  # Expected by replay buffer

  def test_image_wrapper_returns_image_observation_on_step(self):
    env = deepmind_control_lib.create_deepmind_control_environment(
        'cartpole', 'swingup', use_image_observations=True)

    env.reset()
    state, _, _, _ = env.step(env.action_space.sample())

    self.assertIsInstance(state, np.ndarray)
    self.assertEqual(state.shape, (84, 84, 3))
    self.assertEqual(state.dtype, np.uint8)  # Expected by replay buffer

  @parameterized.named_parameters(
      dict(testcase_name='0_to_0', fill_value=0.0, expected=0),
      dict(testcase_name='1_to_255', fill_value=1.0, expected=255))
  def test_float_image_observation_is_scaled_correctly_to_uint(
      self, fill_value: float, expected: int):
    env = MockDeepmindControlSuiteEnvironment(img_dtype=np.float32,
                                              img_fill_value=fill_value)
    env = deepmind_control_lib.DeepmindControlWithImagesPreprocessing(env)

    state = env.reset()

    np.testing.assert_equal(state,
                            np.full_like(state, expected, dtype=np.uint8))

  def test_action_repeats_successfully_apply_repeated_action(self):
    action_repeat = 5
    control_env = self.create_env()
    gym_env = deepmind_control_lib.DeepmindControlPreprocessing(
        control_env, action_repeat=action_repeat)

    gym_env.reset()

    # Step the environment until it is two steps away from being a game over.
    for _ in range(self.max_steps // action_repeat - 2):
      gym_env.step(gym_env.action_space.sample())
    gym_env.step(gym_env.action_space.sample())
    is_penultimate_game_over = gym_env.game_over
    gym_env.step(gym_env.action_space.sample())
    is_final_game_over = gym_env.game_over

    # We will only apply the required number of steps to reach a game over
    # if actions are being repeated as specified.
    self.assertFalse(is_penultimate_game_over)
    self.assertTrue(is_final_game_over)

  def test_action_repeats_obey_end_of_episode(self):
    action_repeat = 7
    total_steps = 10
    step_return_values = [_MID_TIMESTEP for _ in range(total_steps - 1)]
    step_return_values.append(LAST_TIMESTEP)

    control_env = self.create_env()
    control_env.step = mock.MagicMock(side_effect=step_return_values)

    gym_env = deepmind_control_lib.DeepmindControlPreprocessing(
        control_env, action_repeat=action_repeat)
    gym_env.reset()

    gym_env.step(gym_env.action_space.sample())
    is_game_over_step1 = gym_env.game_over

    gym_env.step(gym_env.action_space.sample())
    is_game_over_step2 = gym_env.game_over

    self.assertFalse(is_game_over_step1)
    self.assertTrue(is_game_over_step2)
    self.assertEqual(control_env.step.call_count, total_steps)


if __name__ == '__main__':
  absltest.main()

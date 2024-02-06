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
"""Deepmind Control Suite utilities.

Includes a wrapper class around Deepmind Control Suite environments.
This class makes general Deepmind Control Suite environments conformant with
the API Dopamine is expecting.
"""

import collections
from typing import Any, Mapping, Tuple

from dm_control import suite
from dm_control.rl import control
import dm_env
from dm_env import specs
import gin
import gym
from gym import spaces
import numpy as np


@gin.configurable
def create_deepmind_control_environment(
    domain_name: str = gin.REQUIRED,
    task_name: str = gin.REQUIRED,
    use_image_observations: bool = gin.REQUIRED,
) -> gym.Env:
  """Wraps a Deepmind Control Suite environment with some basic preprocessing.

  Args:
    domain_name: The name of the environment to run.
    task_name: The name of the task to run.
    use_image_observations: If True, the created environment will return image
      observations. Else, it will return the state vector.

  Returns:
    A Gym environment with some standard preprocessing.
  """
  env = suite.load(
      domain_name, task_name, environment_kwargs={'flat_observation': True}
  )

  # Wrap the returned environment in a class which conforms to the API expected
  # by Dopamine.
  if use_image_observations:
    env = DeepmindControlWithImagesPreprocessing(env)
  else:
    env = DeepmindControlPreprocessing(env)

  return env


@gin.configurable(allowlist=['action_repeat'])
class DeepmindControlPreprocessing(gym.Env):
  """A DM Control Suite preprocessing wrapper.

  Attributes:
    environment: The underlying environment that is wrapped for preprocessing.
    game_over: True when the last environment step led to a game over signal.
  """

  def __init__(self, environment: control.Environment, action_repeat: int = 1):
    """Initializes a preprocessing wrapper for a deepmind control environment.

    Args:
      environment: The environment to wrap for preprocessing.
      action_repeat: The number of times to repeat a supplied action. Must be
        greater than 0.
    """
    if action_repeat < 1:
      raise ValueError(f'Action repeat must be > 0. Got {action_repeat}.')

    self.environment = environment
    self._action_repeat = action_repeat
    self.game_over = False

  @property
  def observation_space(self) -> spaces.Box:
    """The observation space of the processed environment."""
    obs_spec = self.environment.observation_spec()
    assert isinstance(obs_spec, collections.OrderedDict)
    # Since observations are flattened, all shapes are 1 dimensional
    low = list()
    high = list()
    for k in obs_spec:
      assert isinstance(obs_spec[k], specs.Array)
      if isinstance(obs_spec[k], specs.BoundedArray):
        low.extend(obs_spec[k].minimum.tolist())
        high.extend(obs_spec[k].maximum.tolist())
      else:
        low.extend([-float('inf') for _ in range(obs_spec[k].shape[0])])
        high.extend([float('inf') for _ in range(obs_spec[k].shape[0])])
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    return spaces.Box(low=low, high=high, dtype=np.float32)

  @property
  def action_space(self) -> spaces.Box:
    """The action space for the processed environment."""
    action_spec = self.environment.action_spec()
    return spaces.Box(
        low=action_spec.minimum, high=action_spec.maximum, dtype=np.float32
    )

  @property
  def reward_range(self) -> Tuple[float, float]:
    """The reward range for the processed environment."""
    return (-float('inf'), float('inf'))

  @property
  def metadata(self) -> Mapping[Any, Any]:
    """The metadata for the processed environment."""
    return {}

  def reset(self) -> np.ndarray:
    """Resets the environment.

    Returns:
      An initial observation.
    """
    timestep = self.environment.reset()
    return self._get_observation(timestep)

  def step(
      self, action: np.ndarray
  ) -> Tuple[np.ndarray, float, bool, Mapping[Any, Any]]:
    """Steps the environment.

    Args:
      action: The action to apply.

    Returns:
      An (observation, reward, is_terminal, info) tuple. For more information,
      see base class.
    """
    accumulated_reward = 0.0
    self.game_over = False
    done = False

    for _ in range(self._action_repeat):
      timestep = self.environment.step(action)
      accumulated_reward += timestep.reward

      if timestep.last():
        self.game_over = True  # Signals end of episode.

        # Only set as terminal if not the result of a timeout.
        done = timestep.discount < 1.0
        break

    observation = self._get_observation(timestep)

    return observation, accumulated_reward, done, {}

  def _get_observation(self, timestep: dm_env.TimeStep) -> np.ndarray:
    return np.concatenate(
        [timestep.observation[k] for k in timestep.observation]
    )


class DeepmindControlWithImagesPreprocessing(DeepmindControlPreprocessing):
  """A DM Control Suite preprocessing wrapper for image observations."""

  def __init__(
      self,
      env: control.Environment,
      observation_shape: Tuple[int, int] = (84, 84),
  ):
    """Constructor for preprocessing wrapper.

    Args:
      env: The environment to wrap.
      observation_shape: The size to reshape the images to. This corresponds to
        (height, width). The output shape will be (height, width, 3), with 3
        corresponding to RGB channels.
    """
    super(DeepmindControlWithImagesPreprocessing, self).__init__(env)

    self._shape = observation_shape

  @property
  def observation_space(self) -> spaces.Box:
    return spaces.Box(
        low=0,
        high=255,
        shape=(
            *self._shape,
            3,
        ),
        dtype=np.uint8,
    )

  def _get_observation(self, timestep: dm_env.TimeStep) -> np.ndarray:
    return self._render_image()

  def _render_image(self) -> np.ndarray:
    """Renders the environment and processes the image.

    The processing steps are:
    1. Convert the image to uint8, as expected by Dopamine's replay buffer.
    2. Resize the image to the specified size.

    Returns:
      An np.ndarray with the image data in rgb format.
    """
    image = self.environment.physics.render(*self._shape, camera_id=0)

    if image.dtype != np.uint8:
      if image.dtype in (np.float32, np.float64):
        image = (image * 255).astype(np.uint8)
      else:
        raise ValueError('Unsupported dtype: {}'.format(image.dtype))

    return image

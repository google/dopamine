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
"""Common utilities for CALE agents."""

import numpy as np


# Take from common/Constants.cpp
_ACTION_STR_TO_INT = {
    'PLAYER_A_NOOP': 0,
    'PLAYER_A_FIRE': 1,
    'PLAYER_A_UP': 2,
    'PLAYER_A_RIGHT': 3,
    'PLAYER_A_LEFT': 4,
    'PLAYER_A_DOWN': 5,
    'PLAYER_A_UPRIGHT': 6,
    'PLAYER_A_UPLEFT': 7,
    'PLAYER_A_DOWNRIGHT': 8,
    'PLAYER_A_DOWNLEFT': 9,
    'PLAYER_A_UPFIRE': 10,
    'PLAYER_A_RIGHTFIRE': 11,
    'PLAYER_A_LEFTFIRE': 12,
    'PLAYER_A_DOWNFIRE': 13,
    'PLAYER_A_UPRIGHTFIRE': 14,
    'PLAYER_A_UPLEFTFIRE': 15,
    'PLAYER_A_DOWNRIGHTFIRE': 16,
    'PLAYER_A_DOWNLEFTFIRE': 17,
    'PLAYER_B_NOOP': 18,
    'PLAYER_B_FIRE': 19,
    'PLAYER_B_UP': 20,
    'PLAYER_B_RIGHT': 21,
    'PLAYER_B_LEFT': 22,
    'PLAYER_B_DOWN': 23,
    'PLAYER_B_UPRIGHT': 24,
    'PLAYER_B_UPLEFT': 25,
    'PLAYER_B_DOWNRIGHT': 26,
    'PLAYER_B_DOWNLEFT': 27,
    'PLAYER_B_UPFIRE': 28,
    'PLAYER_B_RIGHTFIRE': 29,
    'PLAYER_B_LEFTFIRE': 30,
    'PLAYER_B_DOWNFIRE': 31,
    'PLAYER_B_UPRIGHTFIRE': 32,
    'PLAYER_B_UPLEFTFIRE': 33,
    'PLAYER_B_DOWNRIGHTFIRE': 34,
    'PLAYER_B_DOWNLEFTFIRE': 35,
    '__invalid__36': 36,
    '__invalid__37': 37,
    '__invalid__38': 38,
    '__invalid__39': 39,
    'RESET': 40,
    'UNDEFINED': 41,
    'RANDOM': 42,
}


def _polar_to_cartesian(r, theta):
  return (r * np.cos(theta), r * np.sin(theta))


def _polar_to_discrete_action(r, theta, fire, threshold=0.5):
  """Convert actions from polar to discrete (strings)."""
  x, y = _polar_to_cartesian(r, theta)
  action = ''
  if y > threshold:
    action = 'UP'
  elif y < -threshold:
    action = 'DOWN'
  if x > threshold:
    action += 'RIGHT'
  elif x < -threshold:
    action += 'LEFT'
  if fire > threshold:
    action += 'FIRE'
  if not action:
    action = 'NOOP'
  return 'PLAYER_A_' + action


def get_action_number(r, theta, fire, threshold=0.5):
  return _ACTION_STR_TO_INT[
      _polar_to_discrete_action(r, theta, fire, threshold)
  ]

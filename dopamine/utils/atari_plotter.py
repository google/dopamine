# coding=utf-8
# Copyright 2018 The Dopamine Authors.
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
"""AtariPlotter used for rendering Atari 2600 frames.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from dopamine.utils import plotter
import gin
import numpy as np
import pygame


@gin.configurable
class AtariPlotter(plotter.Plotter):
  """A Plotter for rendering Atari 2600 frames."""

  _defaults = {
      'x': 0,
      'y': 0,
      'input_width': 160,
      'input_height': 210,
      'width': 160,
      'height': 210,
  }

  def __init__(self, parameter_dict=None):
    """Constructor for AtariPlotter.

    Args:
      parameter_dict: None or dict of parameter specifications for
        visualization. If an expected parameter is present, its value will
        be used, otherwise it will use defaults.
    """
    super(AtariPlotter, self).__init__(parameter_dict)
    assert 'environment' in self.parameters
    self.game_surface = pygame.Surface((self.parameters['input_width'],
                                        self.parameters['input_height']))

  def draw(self):
    """Render the Atari 2600 frame.

    Returns:
      object to be rendered by AgentVisualizer.
    """
    environment = self.parameters['environment']
    numpy_surface = np.frombuffer(self.game_surface.get_buffer(),
                                  dtype=np.int32)
    obs = environment.render(mode='rgb_array').astype(np.int32)
    obs = np.transpose(obs)
    obs = np.swapaxes(obs, 1, 2)
    obs = obs[2] | (obs[1] << 8) | (obs[0] << 16)
    np.copyto(numpy_surface, obs.ravel())
    return pygame.transform.scale(self.game_surface,
                                  (self.parameters['width'],
                                   self.parameters['height']))

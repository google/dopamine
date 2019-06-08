# coding=utf-8
"""AtariPlotter used for rendering Atari 2600 frames.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
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
    self.game_surface = pygame.Surface((self.parameters['width'],
                                        self.parameters['height']))

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

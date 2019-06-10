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
"""BarPlotter used for drawing bar plots.

Note that a side effect of using this class is to change the font used by
matplotlib. Unless you're planning to use matplotlib elsewhere in your code,
this should be a non-issue.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from dopamine.utils import plotter
import gin
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pygame


# You can change this to use your own palette. A site with great examples is:
# https://www.dtelepathy.com/blog/inspiration/24-flat-designs-with-compelling-color-palettes
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


@gin.configurable
class BarPlotter(plotter.Plotter):
  """A Plotter for generating bar plots."""

  _defaults = {
      'x': 0,
      'y': 0,
      'width': 213,
      'height': 210,
      'fontsize': 30,
      'bg_color': '#f8f7f2',
      'face_color': '#ffffff',
      'colors': COLORS,
      'max_width': 500,
      'figsize': (12, 9),
      'font': {'family': 'Bitstream Vera Sans',
               'weight': 'regular',
               'size': 26},
  }

  def __init__(self, parameter_dict=None):
    """Constructor for BarPlotter.

    This expects a callable 'get_bar_data_fn' in the parameters, which will
    return a list of distributions for each of the actions.  Typically, this
    will be a callback from the agent, which will return some useful information
    about its performance.

    Args:
      parameter_dict: None or dict of parameter specifications for
        visualization. If an expected parameter is present, its value will
        be used, otherwise it will use defaults.
    """
    super(BarPlotter, self).__init__(parameter_dict)
    assert 'get_bar_data_fn' in self.parameters
    self.fig = plt.figure(frameon=False, figsize=self.parameters['figsize'])
    self.plot = self.fig.add_subplot(111)
    self.plot_surface = None
    # This sets the font used by the plotter to be the desired font.
    matplotlib.rc('font', **self.parameters['font'])

  def draw(self):
    """Draw the bar plot.

    If `parameter_dict` contains a 'legend' key pointing to a list of labels,
    this will be used as the legend labels in the plot.

    Returns:
      object to be rendered by AgentVisualizer.
    """
    self._setup_plot()
    num_colors = len(self.parameters['colors'])
    bar_data = self.parameters['get_bar_data_fn']()
    num_actions, num_bins = bar_data.shape
    for i in range(num_actions):
      self.plot.bar(np.arange(num_bins), bar_data[i],
                    color=self.parameters['colors'][i % num_colors])
    if 'legend' in self.parameters:
      self.plot.legend(self.parameters['legend'])
    self.fig.canvas.draw()
    # Now transfer to surface.
    width, height = self.fig.canvas.get_width_height()
    if self.plot_surface is None:
      self.plot_surface = pygame.Surface((width, height))
    plot_buffer = np.frombuffer(self.fig.canvas.buffer_rgba(), np.uint32)
    surf_buffer = np.frombuffer(self.plot_surface.get_buffer(),
                                dtype=np.int32)
    np.copyto(surf_buffer, plot_buffer)
    return pygame.transform.smoothscale(
        self.plot_surface,
        (self.parameters['width'], self.parameters['height']))

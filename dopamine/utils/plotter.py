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
"""Base class for plotters.

This class provides the core functionality for Plotter objects. Specifically, it
initializes `self.parameters` with the values passed through the constructor or
with the provided defaults (specified in each child class), and specifies the
abstract `draw()` method, which child classes will need to implement.

This class also provides a helper function `_setup_plot` for Plotters based on
matplotlib.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class Plotter(object):
  """Abstract base class for plotters."""
  __metaclass__ = abc.ABCMeta

  def __init__(self, parameter_dict=None):
    """Constructor for a Plotter, each child class must define _defaults.

    It will ensure there are values for 'x' and 'y' in `self.parameters`. The
    other key/values will come from either `parameter_dict` or, if not specified
    there, from `self._defaults`.

    Args:
      parameter_dict: None or dict of parameter specifications for
        visualization. If an expected parameter is present, its value will
        be used, otherwise it will use defaults.
    """
    self.parameters = {'x': 0, 'y': 0}
    self.parameters.update(self._defaults)
    self.parameters.update(parameter_dict)

  def _setup_plot(self):
    """Helpful common functionality when rendering matplotlib-style plots."""
    self.plot.cla()  # Clear current figure.
    self.fig.patch.set_facecolor(self.parameters['face_color'])
    try:
      self.plot.set_facecolor(self.parameters['bg_color'])
    except AttributeError:
      self.plot.set_axis_bgcolor(self.parameters['bg_color'])
    if 'xlabel' in self.parameters:
      self.plot.set_xlabel(self.parameters['xlabel'],
                           fontsize=self.parameters['fontsize'] - 2)
    if 'ylabel' in self.parameters:
      self.plot.set_ylabel(self.parameters['ylabel'],
                           fontsize=self.parameters['fontsize'] - 2)
    if 'title' in self.parameters:
      self.plot.set_title(self.parameters['title'],
                          fontsize=self.parameters['fontsize'] + 2)
    if 'xticks' in self.parameters:
      self.plot.set_xticks(self.parameters['xticks'])
    if 'xticklabels' in self.parameters:
      self.plot.set_xticklabels(self.parameters['xticklabels'])

    self.plot.tick_params(labelsize=self.parameters['fontsize'])

  @abc.abstractmethod
  def draw(self):
    """Draw a plot.

    Returns:
      object to be rendered by AgentVisualizer.
    """
    pass

  @property
  def x(self):
    return self.parameters['x']

  @property
  def y(self):
    return self.parameters['y']

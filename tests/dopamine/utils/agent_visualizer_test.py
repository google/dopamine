# coding=utf-8
# Copyright 2019 The Dopamine Authors.
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
"""Tests for dopamine.utils.agent_visualizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil



from absl import flags
from dopamine.utils.agent_visualizer import AgentVisualizer
from dopamine.utils.line_plotter import LinePlotter
import numpy as np
from PIL import Image
import tensorflow as tf


FLAGS = flags.FLAGS


class AgentVisualizerTest(tf.test.TestCase):

  def setUp(self):
    super(AgentVisualizerTest, self).setUp()
    self._test_subdir = os.path.join('/tmp/dopamine_tests', 'agent_visualizer')
    shutil.rmtree(self._test_subdir, ignore_errors=True)
    os.makedirs(self._test_subdir)

  def test_agent_visualizer_save_frame(self):
    parameter_dict = LinePlotter._defaults.copy()
    parameter_dict['get_line_data_fn'] = lambda: [[1, 2, 3]]
    plotter = LinePlotter(parameter_dict=parameter_dict)

    agent_visualizer = AgentVisualizer(self._test_subdir, [plotter])
    agent_visualizer.save_frame()

    frame_filename = os.path.join(self._test_subdir, 'frame_000000.png')
    self.assertTrue(tf.io.gfile.exists(frame_filename))

    im = Image.open(frame_filename)
    im_arr = np.array(im)
    self.assertTrue(np.array_equal(im_arr, agent_visualizer.record_frame))

if __name__ == '__main__':
  tf.compat.v1.disable_v2_behavior()
  tf.test.main()

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
"""Code to visualize different aspects of an agent's behaviour.

This file defines the class AgentVisualizer, which allows one to combine
a number of Plotter objects into a series of single images, generated during
agent interaction with the environment.
If requested, this class will combine the image files into a movie.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess



import gin
import numpy as np
from PIL import Image
import pygame


@gin.configurable
class AgentVisualizer(object):
  """Code to visualize an agent's behaviour."""

  def __init__(
      self,
      record_path,
      plotters,
      screen_width=160,
      screen_height=210,
      render_rate=1,
      file_types=('png', ''),
      filename_format='frame_{:06d}',
  ):
    """Constructor for the AgentVisualizer class.

    This class generates a series of images built by a set of Plotters. These
    images are then saved to disk.

    It can optionally generate a video by concatenating all the images with
    ffmpeg.

    Args:
      record_path: str, path where to save files.
      plotters: list of `Plotter` objects to draw.
      screen_width: int, width of generated images.
      screen_height: int, height of generated images.
      render_rate: int, frame frequency at which to generate files.
      file_types: list of str, specifies the file types to generate.
      filename_format: str, format to use for saving files.
    """
    self.record_path = record_path
    self.plotters = plotters
    self.screen_width = screen_width
    self.screen_height = screen_height
    self.render_rate = render_rate
    self.file_types = file_types
    self.filename_format = filename_format
    self.step = 0
    self.record_frame = np.zeros(
        (self.screen_height, self.screen_width, 3), dtype=np.uint8
    )
    # This is necessary to avoid a `pygame.error: No available video device`
    # error.
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    pygame.init()
    self.screen = pygame.display.set_mode(
        (self.screen_width, self.screen_height), 0, 32
    )

  def visualize(self):
    if self.step % self.render_rate == 0:
      self.screen.fill((0, 0, 0))
      for plotter in self.plotters:
        self.screen = self.screen.copy()  # To avoid locked Surfaces issue.
        self.screen.blit(plotter.draw(), (plotter.x, plotter.y))
      self.save_frame()
    self.step += 1

  def save_frame(self):
    """Save a frame to disk and generate a video, if enabled."""
    screen_buffer = np.frombuffer(
        self.screen.get_buffer(), dtype=np.int32
    ).reshape(self.screen_height, self.screen_width)
    sb = screen_buffer[:, 0 : self.screen_width]
    self.record_frame[..., 2] = sb % 256
    self.record_frame[..., 1] = (sb >> 8) % 256
    self.record_frame[..., 0] = (sb >> 16) % 256
    frame_number = self.step // self.render_rate
    for file_type in self.file_types:
      if not file_type:
        continue
      filename = self.filename_format.format(frame_number) + '.{}'.format(
          file_type
      )
      im = Image.fromarray(self.record_frame)
      im.save(os.path.join(self.record_path, filename))

  def generate_video(self, video_file='video.mp4'):
    """Generates a video, requires 'png' be in file_types.

    Note that this will issue a `subprocess.call` to `ffmpeg`, so only use this
    functionality with trusted paths.

    Args:
      video_file: str, name of video file to generate.
    """
    if 'png' not in self.file_types:
      return
    os.chdir(self.record_path)
    file_regex = self.filename_format.replace('{:', '%').replace('}', '')
    file_regex += '.png'
    subprocess.call([
        'ffmpeg',
        '-r',
        '30',
        '-f',
        'image2',
        '-s',
        '1920x1080',
        '-i',
        file_regex,
        '-vcodec',
        'libx264',
        '-crf',
        '25',
        '-pix_fmt',
        'yuv420p',
        video_file,
    ])

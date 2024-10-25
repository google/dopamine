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
"""Library used by example_viz.py to generate visualizations.

This file illustrates the following:
  - How to subclass an existing agent to add visualization functionality.
    - For DQN we visualize the cumulative rewards and the Q-values for each
      action (MyDQNAgent).
    - For Rainbow we visualize the cumulative rewards and the Q-value
      distributions for each action (MyRainbowAgent).
  - How to subclass Runner to run in eval mode, lay out the different subplots,
    generate the visualizations, and compile them into a video (MyRunner).
  - The function `run()` is the main entrypoint for running everything.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import logging

from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment
from dopamine.tf.agents.dqn import dqn_agent
from dopamine.tf.agents.rainbow import rainbow_agent
from dopamine.utils import agent_visualizer
from dopamine.utils import atari_plotter
from dopamine.utils import bar_plotter
from dopamine.utils import line_plotter
import gin
import numpy as np
import tensorflow as tf
import tf_slim


class MyDQNAgent(dqn_agent.DQNAgent):
  """Sample DQN agent to visualize Q-values and rewards."""

  def __init__(self, sess, num_actions, summary_writer=None):
    super(MyDQNAgent, self).__init__(
        sess, num_actions, summary_writer=summary_writer
    )
    self.q_values = [[] for _ in range(num_actions)]
    self.rewards = []

  def step(self, reward, observation):
    self.rewards.append(reward)
    return super(MyDQNAgent, self).step(reward, observation)

  def _select_action(self):
    action = super(MyDQNAgent, self)._select_action()
    q_vals = self._sess.run(
        self._net_outputs.q_values, {self.state_ph: self.state}
    )[0]
    for i in range(len(q_vals)):
      self.q_values[i].append(q_vals[i])
    return action

  def reload_checkpoint(self, checkpoint_path, use_legacy_checkpoint=False):
    if use_legacy_checkpoint:
      variables_to_restore = atari_lib.maybe_transform_variable_names(
          tf.compat.v1.global_variables(), legacy_checkpoint_load=True
      )
    else:
      global_vars = set([x.name for x in tf.compat.v1.global_variables()])
      ckpt_vars = [
          '{}:0'.format(name)
          for name, _ in tf.train.list_variables(checkpoint_path)
      ]
      include_vars = list(global_vars.intersection(set(ckpt_vars)))
      variables_to_restore = tf_slim.get_variables_to_restore(
          include=include_vars
      )
    if variables_to_restore:
      reloader = tf.compat.v1.train.Saver(var_list=variables_to_restore)
      reloader.restore(self._sess, checkpoint_path)
      logging.info('Done restoring from %s', checkpoint_path)
    else:
      logging.info('Nothing to restore!')

  def get_q_values(self):
    return self.q_values

  def get_rewards(self):
    return [np.cumsum(self.rewards)]


class MyRainbowAgent(rainbow_agent.RainbowAgent):
  """Sample Rainbow agent to visualize Q-values and rewards."""

  def __init__(self, sess, num_actions, summary_writer=None):
    super(MyRainbowAgent, self).__init__(
        sess, num_actions, summary_writer=summary_writer
    )
    self.rewards = []

  def step(self, reward, observation):
    self.rewards.append(reward)
    return super(MyRainbowAgent, self).step(reward, observation)

  def reload_checkpoint(self, checkpoint_path, use_legacy_checkpoint=False):
    if use_legacy_checkpoint:
      variables_to_restore = atari_lib.maybe_transform_variable_names(
          tf.compat.v1.global_variables(), legacy_checkpoint_load=True
      )
    else:
      global_vars = set([x.name for x in tf.compat.v1.global_variables()])
      ckpt_vars = [
          '{}:0'.format(name)
          for name, _ in tf.train.list_variables(checkpoint_path)
      ]
      include_vars = list(global_vars.intersection(set(ckpt_vars)))
      variables_to_restore = tf_slim.get_variables_to_restore(
          include=include_vars
      )
    if variables_to_restore:
      reloader = tf.compat.v1.train.Saver(var_list=variables_to_restore)
      reloader.restore(self._sess, checkpoint_path)
      logging.info('Done restoring from %s', checkpoint_path)
    else:
      logging.info('Nothing to restore!')

  def get_probabilities(self):
    return self._sess.run(
        tf.squeeze(self._net_outputs.probabilities), {self.state_ph: self.state}
    )

  def get_rewards(self):
    return [np.cumsum(self.rewards)]


class MyRunner(run_experiment.Runner):
  """Sample Runner class to generate visualizations."""

  def __init__(
      self,
      base_dir,
      trained_agent_ckpt_path,
      create_agent_fn,
      use_legacy_checkpoint=False,
  ):
    self._trained_agent_ckpt_path = trained_agent_ckpt_path
    self._use_legacy_checkpoint = use_legacy_checkpoint
    super(MyRunner, self).__init__(base_dir, create_agent_fn)

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    self._agent.reload_checkpoint(
        self._trained_agent_ckpt_path, self._use_legacy_checkpoint
    )
    self._start_iteration = 0

  def _run_one_iteration(self, iteration):
    statistics = iteration_statistics.IterationStatistics()
    logging.info('Starting iteration %d', iteration)
    _, _ = self._run_eval_phase(statistics)
    return statistics.data_lists

  def visualize(self, record_path, num_global_steps=500):
    if not tf.io.gfile.exists(record_path):
      tf.io.gfile.makedirs(record_path)
    self._agent.eval_mode = True

    # Set up the game playback rendering.
    atari_params = {'environment': self._environment}
    atari_plot = atari_plotter.AtariPlotter(parameter_dict=atari_params)
    # Plot the rewards received next to it.
    reward_params = {
        'x': atari_plot.parameters['width'],
        'xlabel': 'Timestep',
        'ylabel': 'Reward',
        'title': 'Rewards',
        'get_line_data_fn': self._agent.get_rewards,
    }
    reward_plot = line_plotter.LinePlotter(parameter_dict=reward_params)
    action_names = [
        'Action {}'.format(x) for x in range(self._agent.num_actions)
    ]
    # Plot Q-values (DQN) or Q-value distributions (Rainbow).
    q_params = {
        'x': atari_plot.parameters['width'] // 2,
        'y': atari_plot.parameters['height'],
        'legend': action_names,
    }
    if 'DQN' in self._agent.__class__.__name__:
      q_params['xlabel'] = 'Timestep'
      q_params['ylabel'] = 'Q-Value'
      q_params['title'] = 'Q-Values'
      q_params['get_line_data_fn'] = self._agent.get_q_values
      q_plot = line_plotter.LinePlotter(parameter_dict=q_params)
    elif 'Implicit' in self._agent.__class__.__name__:
      q_params['xlabel'] = 'Timestep'
      q_params['ylabel'] = 'Quantile Value'
      q_params['title'] = 'Quantile Values'
      q_params['get_line_data_fn'] = self._agent.get_q_values
      q_plot = line_plotter.LinePlotter(parameter_dict=q_params)
    else:
      q_params['xlabel'] = 'Return'
      q_params['ylabel'] = 'Return probability'
      q_params['title'] = 'Return distribution'
      q_params['get_bar_data_fn'] = self._agent.get_probabilities
      q_plot = bar_plotter.BarPlotter(parameter_dict=q_params)
    screen_width = (
        atari_plot.parameters['width'] + reward_plot.parameters['width']
    )
    screen_height = (
        atari_plot.parameters['height'] + q_plot.parameters['height']
    )
    # Dimensions need to be divisible by 2:
    if screen_width % 2 > 0:
      screen_width += 1
    if screen_height % 2 > 0:
      screen_height += 1
    visualizer = agent_visualizer.AgentVisualizer(
        record_path=record_path,
        plotters=[atari_plot, reward_plot, q_plot],
        screen_width=screen_width,
        screen_height=screen_height,
    )
    global_step = 0
    while global_step < num_global_steps:
      initial_observation = self._environment.reset()
      action = self._agent.begin_episode(initial_observation)
      while True:
        observation, reward, is_terminal, _ = self._environment.step(action)
        global_step += 1
        visualizer.visualize()
        if self._environment.game_over or global_step >= num_global_steps:
          break
        elif is_terminal:
          self._agent.end_episode(reward)
          action = self._agent.begin_episode(observation)
        else:
          action = self._agent.step(reward, observation)
      self._end_episode(reward)
    visualizer.generate_video()


def create_dqn_agent(sess, environment, summary_writer=None):
  return MyDQNAgent(
      sess,
      num_actions=environment.action_space.n,
      summary_writer=summary_writer,
  )


def create_rainbow_agent(sess, environment, summary_writer=None):
  return MyRainbowAgent(
      sess,
      num_actions=environment.action_space.n,
      summary_writer=summary_writer,
  )


def create_runner(
    base_dir, trained_agent_ckpt_path, agent='dqn', use_legacy_checkpoint=False
):
  create_agent = create_dqn_agent if agent == 'dqn' else create_rainbow_agent
  return MyRunner(
      base_dir, trained_agent_ckpt_path, create_agent, use_legacy_checkpoint
  )


def run(
    agent, game, num_steps, root_dir, restore_ckpt, use_legacy_checkpoint=False
):
  """Main entrypoint for running and generating visualizations.

  Args:
    agent: str, agent type to use.
    game: str, Atari 2600 game to run.
    num_steps: int, number of steps to play game.
    root_dir: str, root directory where files will be stored.
    restore_ckpt: str, path to the checkpoint to reload.
    use_legacy_checkpoint: bool, whether to restore from a legacy (pre-Keras)
      checkpoint.
  """
  tf.compat.v1.reset_default_graph()
  config = """
  atari_lib.create_atari_environment.game_name = '{}'
  WrappedReplayBuffer.replay_capacity = 300
  """.format(game)
  base_dir = os.path.join(root_dir, 'agent_viz', game, agent)
  gin.parse_config(config)
  runner = create_runner(base_dir, restore_ckpt, agent, use_legacy_checkpoint)
  runner.visualize(os.path.join(base_dir, 'images'), num_global_steps=num_steps)

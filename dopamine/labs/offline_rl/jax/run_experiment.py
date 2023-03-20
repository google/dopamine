# coding=utf-8
# Copyright 2023 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runner for experiments with a fixed replay buffer."""

import sys
import time

from absl import logging
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment
from dopamine.metrics import statistics_instance
import gin
import tensorflow as tf


@gin.configurable
class FixedReplayRunner(run_experiment.Runner):
  """Runner for Dopamine experiments with fixed replay buffer."""

  def __init__(self, base_dir, create_agent_fn, num_epochs=None):
    super().__init__(base_dir, create_agent_fn)
    if num_epochs is not None:
      try:
        # pylint: disable=protected-access
        gradients_per_epoch = self._agent._replay.gradient_steps_per_epoch
        # Each intermediate evaluation happens every epoch // 3
        self._training_steps = gradients_per_epoch // 3
        # pylint: enable=protected-access
        self._num_iterations = num_epochs * 3
        logging.info('\t Num epochs: %d', num_epochs)
        logging.info('\t Training steps: %d', self._training_steps)
        logging.info('\t Num iterations: %d', self._num_iterations)
      except AttributeError:
        pass

  def _run_train_phase(self):
    """Run training phase."""
    self._agent.eval_mode = False
    start_time = time.time()

    for i in range(self._training_steps):
      if i % 100 == 0:
        # We use sys.stdout.write instead of logging so as to flush frequently
        # without generating a line break.
        sys.stdout.write('Training step: {}/{}\r'.format(
            i, self._training_steps))
        sys.stdout.flush()

      self._agent.train_step()

    time_delta = time.time() - start_time
    logging.info('Average training steps per second: %.2f',
                 self._training_steps / time_delta)

  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction."""
    statistics = iteration_statistics.IterationStatistics()
    logging.info('Starting iteration %d', iteration)
    # Reload the replay buffer at every iteration
    self._agent.reload_data()  # pylint: disable=protected-access
    self._run_train_phase()

    num_episodes_eval, average_reward_eval = self._run_eval_phase(statistics)

    if self._has_collector_dispatcher:
      self._collector_dispatcher.write([
          statistics_instance.StatisticsInstance('Eval/NumEpisodes',
                                                 num_episodes_eval,
                                                 iteration),
          statistics_instance.StatisticsInstance('Eval/AverageReturns',
                                                 average_reward_eval,
                                                 iteration)
      ])

    if self._summary_writer is not None:
      self._save_tensorboard_summaries(iteration, num_episodes_eval,
                                       average_reward_eval)
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration, num_episodes_eval,
                                  average_reward_eval):
    """Save statistics as tensorboard summaries.

    Args:
      iteration: int, The current iteration number.
      num_episodes_eval: int, number of evaluation episodes run.
      average_reward_eval: float, The average evaluation reward.
    """
    with self._summary_writer.as_default():
      tf.summary.scalar('Eval/NumEpisodes', num_episodes_eval,
                        step=iteration)
      tf.summary.scalar('Eval/AverageReturns', average_reward_eval,
                        step=iteration)

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
"""Runner for evaluation based on a fixed number of episodes."""

import sys

from absl import logging
from dopamine.discrete_domains import run_experiment
import gin
import jax


@gin.configurable
class MaxEpisodeEvalRunner(run_experiment.Runner):
  """Runner for evaluating using a fixed number of episodes rather than steps."""

  def __init__(self,
               base_dir,
               create_agent_fn,
               num_eval_episodes=100,
               max_noops=30):
    """Specify the number of evaluation episodes."""
    super().__init__(base_dir, create_agent_fn)
    self._num_eval_episodes = num_eval_episodes
    logging.info('Num evaluation episodes: %d', num_eval_episodes)
    self._evaluation_steps = None
    self._max_noops = max_noops

  def _initialize_episode(self):
    """Initialization for a new episode with random number of no-ops.

    Returns:
     action: int, the initial action chosen by the agent.
    """
    initial_observation = self._environment.reset()
    if self._max_noops > 0:
      initial_observation = self._run_no_ops()
    return self._agent.begin_episode(initial_observation)

  def _run_no_ops(self):
    """Executes `num_noops` no-ops in the environment."""
    self._agent._rng, rng = jax.random.split(self._agent._rng)  # pylint: disable=protected-access
    num_noops = jax.random.randint(
        rng, shape=(), minval=0, maxval=self._max_noops)
    for _ in range(num_noops):
      # Assumes raw action 0 is always no-op
      self._environment.environment.ale.act(0)
    if self._environment.environment.ale.game_over():
      observation = self._environment.reset()
    else:
      observation = self._environment._pool_and_resize()  # pylint: disable=protected-access
    return observation

  def _run_one_phase_fix_episodes(self, max_episodes, statistics, run_mode_str):
    """Runs the agent/environment loop until a desired number of steps.

    We follow the Machado et al., 2017 convention of running full episodes,
    and terminating once we've run a minimum number of steps.

    Args:
      max_episodes: int, fixed number of episodes to generate in this phase.
      statistics: `IterationStatistics` object which records the experimental
        results.
      run_mode_str: str, describes the run mode for this agent.

    Returns:
      Tuple containing the number of steps taken in this phase (int), the sum of
        returns (float), and the number of episodes performed (int).
    """
    step_count = 0
    num_episodes = 0
    sum_returns = 0.

    while num_episodes < max_episodes:
      episode_length, episode_return = self._run_one_episode()
      statistics.append({
          '{}_episode_lengths'.format(run_mode_str): episode_length,
          '{}_episode_returns'.format(run_mode_str): episode_return
      })
      step_count += episode_length
      sum_returns += episode_return
      num_episodes += 1
      # We use sys.stdout.write instead of logging so as to flush frequently
      # without generating a line break.
      sys.stdout.write('Steps executed: {} '.format(step_count) +
                       'Episode length: {} '.format(episode_length) +
                       'Num episodes: {} '.format(num_episodes) +
                       'Return: {}\r'.format(episode_return))
      sys.stdout.flush()
    return step_count, sum_returns, num_episodes

  def _run_eval_phase(self, statistics):
    """Run evaluation phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
    """
    # Perform the evaluation phase -- no learning.
    self._agent.eval_mode = True
    _, sum_returns, num_episodes = self._run_one_phase_fix_episodes(
        self._num_eval_episodes, statistics, 'eval')
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    logging.info('Average undiscounted return per evaluation episode: %.2f',
                 average_return)
    statistics.append({'eval_average_return': average_return})
    return num_episodes, average_return

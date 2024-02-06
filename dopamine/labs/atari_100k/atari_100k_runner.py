# coding=utf-8
# Copyright 2023 The Dopamine Authors.
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
"""Runner for evaluating using a fixed number of episodes."""

import functools
import os
import sys
import time

from absl import logging
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment
from dopamine.labs.atari_100k import normalization_utils
from dopamine.metrics import collector_dispatcher
from dopamine.metrics import statistics_instance
import gin
import jax
import numpy as np
import tensorflow as tf


def create_env_wrapper(create_env_fn):
  def inner_create(*args, **kwargs):
    env = create_env_fn(*args, **kwargs)
    env.cum_length = 0
    env.cum_reward = 0
    return env

  return inner_create


@gin.configurable
class DataEfficientAtariRunner(run_experiment.Runner):
  """Runner for evaluating using a fixed number of episodes rather than steps.

  Also restricts data collection to a strict cap,
  following conventions in data-efficient RL research.
  """

  def __init__(
      self,
      base_dir,
      create_agent_fn,
      game_name=None,
      create_environment_fn=atari_lib.create_atari_environment,
      num_eval_episodes=100,
      max_noops=30,
      parallel_eval=True,
      num_eval_envs=100,
      num_train_envs=1,
      eval_one_to_one=True,
      log_normalized_scores=False,
  ):
    """Specify the number of evaluation episodes."""
    if game_name is not None:
      create_environment_fn = functools.partial(
          create_environment_fn, game_name=game_name
      )
      self.game_name = game_name.lower().replace('_', '').replace(' ', '')

      if log_normalized_scores:
        raise ValueError(
            'Game name must not be None if logging normalized scores.'
        )
    super().__init__(
        base_dir, create_agent_fn, create_environment_fn=create_environment_fn
    )

    self._num_iterations = int(self._num_iterations)
    self._start_iteration = int(self._start_iteration)

    self._num_eval_episodes = num_eval_episodes
    logging.info('Num evaluation episodes: %d', num_eval_episodes)
    self._evaluation_steps = None
    self.num_steps = 0
    self.total_steps = self._training_steps * self._num_iterations
    self.create_environment_fn = create_env_wrapper(create_environment_fn)

    self.max_noops = max_noops
    self.parallel_eval = parallel_eval
    self.num_eval_envs = num_eval_envs
    self.num_train_envs = num_train_envs
    self.eval_one_to_one = eval_one_to_one

    self.train_envs = [
        self.create_environment_fn() for i in range(num_train_envs)
    ]
    self.train_state = None
    self._agent.reset_all(self._initialize_episode(self.train_envs))
    self._agent.cache_train_state()

    self.log_normalized_scores = log_normalized_scores
    # Create a collector dispatcher for metrics reporting.
    self._collector_dispatcher = collector_dispatcher.CollectorDispatcher(
        self._base_dir
    )
    set_collector_dispatcher_fn = getattr(
        self._agent, 'set_collector_dispatcher', None
    )
    if callable(set_collector_dispatcher_fn):
      set_collector_dispatcher_fn(self._collector_dispatcher)

  def _run_one_phase(
      self,
      envs,
      steps,
      max_episodes,
      statistics,
      run_mode_str,
      needs_reset=False,
      one_to_one=False,
      resume_state=None,
  ):
    """Runs the agent/environment loop until a desired number of steps.

    We terminate precisely when the desired number of steps has been reached,
    unlike some other implementations.

    Args:
      envs: environments to use in this phase.
      steps: int, how many steps to run in this phase (or None).
      max_episodes: int, maximum number of episodes to generate in this phase.
      statistics: `IterationStatistics` object which records the experimental
        results.
      run_mode_str: str, describes the run mode for this agent.
      needs_reset: bool, whether to reset all environments before starting.
      one_to_one: bool, whether to precisely match each episode in
        `max_episodes` to an environment in `envs`. True is faster but only
        works in some situations (e.g., evaluation).
      resume_state: bool, whether to have the agent resume its prior state for
        the current mode.

    Returns:
      Tuple containing the number of steps taken in this phase (int), the
      sum of
        returns (float), and the number of episodes performed (int).
    """
    step_count = 0
    num_episodes = 0
    sum_returns = 0.0

    (episode_lengths, episode_returns, state, envs) = self._run_parallel(
        episodes=max_episodes,
        envs=envs,
        one_to_one=one_to_one,
        needs_reset=needs_reset,
        resume_state=resume_state,
        max_steps=steps,
    )

    for episode_length, episode_return in zip(episode_lengths, episode_returns):
      statistics.append({
          '{}_episode_lengths'.format(run_mode_str): episode_length,
          '{}_episode_returns'.format(run_mode_str): episode_return,
      })
      if run_mode_str == 'train':
        # we use one extra frame at the starting
        self.num_steps += episode_length
      step_count += episode_length
      sum_returns += episode_return
      num_episodes += 1
      sys.stdout.flush()
      if self._summary_writer is not None:
        with self._summary_writer.as_default():
          _ = (
              tf.summary.scalar(
                  'train_episode_returns',
                  float(episode_return),
                  step=self.num_steps,
              ),
          )
          _ = tf.summary.scalar(
              'train_episode_lengths',
              float(episode_length),
              step=self.num_steps,
          )
    return step_count, sum_returns, num_episodes, state, envs

  def _initialize_episode(self, envs):
    """Initialization for a new episode.

    Args:
      envs: Environments to initialize episodes for.

    Returns:
      action: int, the initial action chosen by the agent.
    """
    observations = []
    for env in envs:
      initial_observation = env.reset()
      if self.max_noops > 0:
        self._agent._rng, rng = jax.random.split(
            self._agent._rng  # pylint: disable=protected-access
        )
        num_noops = jax.random.randint(rng, (), 0, self.max_noops)
        for _ in range(num_noops):
          initial_observation, _, terminal, _ = env.step(0)
          if terminal:
            initial_observation = env.reset()
      observations.append(initial_observation)
    initial_observation = np.stack(observations, 0)

    return initial_observation

  def _run_parallel(
      self,
      envs,
      episodes=None,
      max_steps=None,
      one_to_one=False,
      needs_reset=True,
      resume_state=None,
  ):
    """Executes a full trajectory of the agent interacting with the environment.

    Args:
      envs: Environments to step in.
      episodes: Optional int, how many episodes to run. Unbounded if None.
      max_steps: Optional int, how many steps to run. Unbounded if None.
      one_to_one: Bool, whether to couple each episode to an environment.
      needs_reset: Bool, whether to reset environments before beginning.
      resume_state: State tuple to resume.

    Returns:
      The number of steps taken and the total reward.
    """
    # You can't ask for 200 episodes run one-to-one on 100 envs
    if one_to_one:
      assert episodes is None or episodes == len(envs)

    # Create envs
    live_envs = list(range(len(envs)))

    if needs_reset:
      new_obs = self._initialize_episode(envs)
      new_obses = np.zeros((2, len(envs), *self._agent.observation_shape, 1))
      self._agent.reset_all(new_obs)

      rewards = np.zeros((len(envs),))
      terminals = np.zeros((len(envs),))
      episode_end = np.zeros((len(envs),))

      cum_rewards = []
      cum_lengths = []
    else:
      assert resume_state is not None
      (new_obses, rewards, terminals, episode_end, cum_rewards, cum_lengths) = (
          resume_state
      )

    total_steps = 0
    total_episodes = 0
    max_steps = np.inf if max_steps is None else max_steps
    step = 0

    # Keep interacting until we reach a terminal state.
    while True:
      live_env_index = 0
      step += 1
      episode_end.fill(0)
      total_steps += len(live_envs)
      actions = self._agent.step()

      # The agent may be hanging on to the previous new_obs, so we shouldn't
      # change it. By alternating, we can make sure we don't end up logging
      # with an offset while limiting memory usage.
      new_obs = new_obses[step % 2]

      # don't want to do a for-loop since live envs may change
      while live_env_index < len(live_envs):
        env_id = live_envs[live_env_index]
        obs, reward, done, _ = envs[env_id].step(actions[live_env_index])
        envs[env_id].cum_length += 1
        envs[env_id].cum_reward += reward
        new_obs[live_env_index] = obs
        rewards[live_env_index] = reward
        terminals[live_env_index] = done

        if (
            envs[env_id].game_over
            or envs[env_id].cum_length == self._max_steps_per_episode
        ):
          total_episodes += 1
          cum_rewards.append(envs[env_id].cum_reward)
          cum_lengths.append(envs[env_id].cum_length)
          envs[env_id].cum_length = 0
          envs[env_id].cum_reward = 0

          log_str = (
              'Steps executed: {} Num episodes: {} Episode length: {}'
              ' Return: {} '.format(
                  total_steps,
                  len(cum_rewards),
                  cum_lengths[-1],
                  cum_rewards[-1],
              )
          )
          logging.info(log_str)
          self._maybe_save_single_summary(
              self.num_steps + total_steps, cum_rewards[-1], cum_lengths[-1]
          )

          if one_to_one:
            new_obses = delete_ind_from_array(new_obses, live_env_index, axis=1)
            new_obs = new_obses[step % 2]
            actions = delete_ind_from_array(actions, live_env_index)
            rewards = delete_ind_from_array(rewards, live_env_index)
            terminals = delete_ind_from_array(terminals, live_env_index)
            self._agent.delete_one(live_env_index)
            del live_envs[live_env_index]
            live_env_index -= 1  # Go back one to make up for deleting a value.
          else:
            episode_end[live_env_index] = 1
            new_obs[live_env_index] = self._initialize_episode([envs[env_id]])
            self._agent.reset_one(env_id=live_env_index)
        elif done:
          self._agent.reset_one(env_id=live_env_index)

        live_env_index += 1

      if self._clip_rewards:
        # Perform reward clipping.
        rewards = np.clip(rewards, -1, 1)

      self._agent.log_transition(
          new_obs, actions, rewards, terminals, episode_end
      )

      if (
          not live_envs
          or (max_steps is not None and total_steps > max_steps)
          or (episodes is not None and total_episodes > episodes)
      ):
        break

    state = (
        new_obses,
        rewards,
        terminals,
        episode_end,
        cum_rewards,
        cum_lengths,
    )
    return cum_lengths, cum_rewards, state, envs

  def _run_train_phase(self, statistics):
    """Run training phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
      average_steps_per_second: float, The average number of steps per
      second.
    """
    # Perform the training phase, during which the agent learns.
    self._agent.eval_mode = False
    self._agent.restore_train_state()
    start_time = time.time()
    (
        number_steps,
        sum_returns,
        num_episodes,
        self.train_state,
        self.train_envs,
    ) = self._run_one_phase(
        self.train_envs,
        self._training_steps,
        max_episodes=None,
        statistics=statistics,
        run_mode_str='train',
        needs_reset=self.train_state is None,
        resume_state=self.train_state,
    )
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    statistics.append({'train_average_return': average_return})
    time_delta = time.time() - start_time
    average_steps_per_second = number_steps / time_delta
    statistics.append(
        {'train_average_steps_per_second': average_steps_per_second}
    )
    logging.info(
        'Average undiscounted return per training episode: %.2f', average_return
    )
    logging.info(
        'Average training steps per second: %.2f', average_steps_per_second
    )

    if self.log_normalized_scores:
      normalized_return = normalization_utils.normalize_score(
          average_return, self.game_name
      )
      statistics.append({'train_average_normalized_score': normalized_return})
      logging.info(
          'Average normalized return per training episode: %.2f',
          normalized_return,
      )
    else:
      normalized_return = None

    self._agent.cache_train_state()
    return (
        num_episodes,
        average_return,
        average_steps_per_second,
        normalized_return,
    )

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
    eval_envs = [
        self.create_environment_fn() for i in range(self.num_eval_envs)
    ]
    _, sum_returns, num_episodes, _, _ = self._run_one_phase(
        eval_envs,
        steps=None,
        max_episodes=self._num_eval_episodes,
        statistics=statistics,
        needs_reset=True,
        resume_state=None,
        one_to_one=self.eval_one_to_one,
        run_mode_str='eval',
    )
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    logging.info(
        'Average undiscounted return per evaluation episode: %.2f',
        average_return,
    )
    statistics.append({'eval_average_return': average_return})
    if self.log_normalized_scores:
      normalized_return = normalization_utils.normalize_score(
          average_return, self.game_name
      )
      statistics.append({'train_average_normalized_score': normalized_return})
      logging.info(
          'Average normalized return per evaluation episode: %.2f',
          normalized_return,
      )
    else:
      normalized_return = None
    return num_episodes, average_return, normalized_return

  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction."""
    statistics = iteration_statistics.IterationStatistics()
    logging.info('Starting iteration %d', iteration)
    (
        num_episodes_train,
        average_reward_train,
        average_steps_per_second,
        norm_score_train,
    ) = self._run_train_phase(statistics)
    num_episodes_eval, average_reward_eval, norm_score_eval = (
        self._run_eval_phase(statistics)
    )
    self._collector_dispatcher.write([
        statistics_instance.StatisticsInstance(
            'Train/NumEpisodes', num_episodes_train, iteration
        ),
        statistics_instance.StatisticsInstance(
            'Train/AverageReturns', average_reward_train, iteration
        ),
        statistics_instance.StatisticsInstance(
            'Train/AverageStepsPerSecond', average_steps_per_second, iteration
        ),
        statistics_instance.StatisticsInstance(
            'Eval/NumEpisodes', num_episodes_eval, iteration
        ),
        statistics_instance.StatisticsInstance(
            'Eval/AverageReturns', average_reward_eval, iteration
        ),
    ])
    self._save_tensorboard_summaries(
        iteration,
        num_episodes_train,
        average_reward_train,
        norm_score_train,
        num_episodes_eval,
        average_reward_eval,
        norm_score_eval,
        average_steps_per_second,
    )
    return statistics.data_lists

  def _maybe_save_single_summary(
      self, iteration, ep_return, length, save_if_eval=False
  ):
    prefix = 'Train/' if not self._agent.eval_mode else 'Eval/'
    if not self._agent.eval_mode or save_if_eval:
      with self._summary_writer.as_default():
        tf.summary.scalar(prefix + 'EpisodeLength', length, step=iteration)
        tf.summary.scalar(prefix + 'EpisodeReturn', ep_return, step=iteration)
        if self.log_normalized_scores:
          normalized_score = normalization_utils.normalize_score(
              ep_return, self.game_name
          )
          tf.summary.scalar(
              prefix + 'EpisodeNormalizedScore',
              normalized_score,
              step=iteration,
          )

  def _save_tensorboard_summaries(
      self,
      iteration,
      num_episodes_train,
      average_reward_train,
      norm_score_train,
      num_episodes_eval,
      average_reward_eval,
      norm_score_eval,
      average_steps_per_second,
  ):
    """Save statistics as tensorboard summaries.

    Args:
      iteration: int, The current iteration number.
      num_episodes_train: int, number of training episodes run.
      average_reward_train: float, The average training reward.
      norm_score_train: float, average training normalized score.
      num_episodes_eval: int, number of evaluation episodes run.
      average_reward_eval: float, The average evaluation reward.
      norm_score_eval: float, average eval normalized score.
      average_steps_per_second: float, The average number of steps per second.
    """
    with self._summary_writer.as_default():
      tf.summary.scalar('Train/NumEpisodes', num_episodes_train, step=iteration)
      tf.summary.scalar(
          'Train/AverageReturns', average_reward_train, step=iteration
      )
      tf.summary.scalar(
          'Train/AverageStepsPerSecond',
          average_steps_per_second,
          step=iteration,
      )
      tf.summary.scalar('Eval/NumEpisodes', num_episodes_eval, step=iteration)
      tf.summary.scalar(
          'Eval/AverageReturns', average_reward_eval, step=iteration
      )
      if self.log_normalized_scores:
        tf.summary.scalar(
            'Train/AverageNormalizedScore', norm_score_train, step=iteration
        )
        tf.summary.scalar(
            'Eval/NormalizedScore', norm_score_eval, step=iteration
        )

  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    logging.info('Beginning training...')
    if self._num_iterations <= self._start_iteration:
      logging.warning(
          'num_iterations (%d) < start_iteration(%d)',
          self._num_iterations,
          self._start_iteration,
      )
      return

    for iteration in range(self._start_iteration, self._num_iterations):
      statistics = self._run_one_iteration(iteration)
      self._log_experiment(iteration, statistics)
      self._checkpoint_experiment(iteration)
    self._summary_writer.flush()


@gin.configurable
class LoggedDataEfficientAtariRunner(DataEfficientAtariRunner):
  """Runner for loading/saving replay data."""

  def __init__(
      self, base_dir, create_agent_fn, load_replay_dir=None, save_replay=False
  ):
    super().__init__(base_dir, create_agent_fn)
    self._load_replay_dir = load_replay_dir
    self._save_replay = save_replay
    logging.info('Load fixed replay from directory: %s', load_replay_dir)
    logging.info('Save replay: %s', save_replay)

  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    if self._load_replay_dir is not None:
      self._agent.load_fixed_replay(self._load_replay_dir)
    super().run_experiment()
    if self._save_replay:
      save_replay_dir = os.path.join(self._base_dir, 'replay_logs')
      self._agent.save_replay(save_replay_dir)


def delete_ind_from_array(array, ind, axis=0):
  start = tuple(([slice(None)] * axis) + [slice(0, ind)])
  end = tuple(([slice(None)] * axis) + [slice(ind + 1, array.shape[axis] + 1)])
  tensor = np.concatenate([array[start], array[end]], axis)
  return tensor

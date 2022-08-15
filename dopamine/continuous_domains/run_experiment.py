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
"""Module defining classes and helper methods for general agents."""

from typing import Optional

from absl import logging
from dopamine.discrete_domains import gym_lib
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment as base_run_experiment
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.agents.sac import sac_agent
# pylint: disable=unused-import
from dopamine.labs.sac_from_pixels import deepmind_control_lib
# pylint: enable=unused-import
from dopamine.metrics import collector_dispatcher
from dopamine.metrics import statistics_instance
from flax.metrics import tensorboard
import gin
from gym import spaces



load_gin_configs = base_run_experiment.load_gin_configs


@gin.configurable
def create_continuous_agent(
    environment: gym_lib.GymPreprocessing,
    agent_name: str,
    summary_writer: Optional[tensorboard.SummaryWriter] = None
) -> dqn_agent.JaxDQNAgent:
  """Creates an agent.

  Args:
    environment:  A gym environment.
    agent_name: str, name of the agent to create.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.

  Returns:
    An RL agent.

  Raises:
    ValueError: If `agent_name` is not in supported list.
  """
  assert agent_name is not None
  if agent_name == 'sac':
    assert isinstance(environment.action_space, spaces.Box)
    assert isinstance(environment.observation_space, spaces.Box)
    return sac_agent.SACAgent(
        action_shape=environment.action_space.shape,
        action_limits=(environment.action_space.low,
                       environment.action_space.high),
        observation_shape=environment.observation_space.shape,
        action_dtype=environment.action_space.dtype,
        observation_dtype=environment.observation_space.dtype,
        summary_writer=summary_writer)
  else:
    raise ValueError(f'Unknown agent: {agent_name}')


@gin.configurable
def create_continuous_runner(base_dir, schedule='continuous_train_and_eval'):
  """Creates an experiment Runner.

  Args:
    base_dir: str, base directory for hosting all subdirectories.
    schedule: string, which type of Runner to use.

  Returns:
    runner: A `Runner` like object.

  Raises:
    ValueError: When an unknown schedule is encountered.
  """
  assert base_dir is not None
  # Continuously runs training and evaluation until max num_iterations is hit.
  if schedule == 'continuous_train_and_eval':
    return ContinuousRunner(base_dir, create_continuous_agent)
  # Continuously runs training until max num_iterations is hit.
  elif schedule == 'continuous_train':
    return ContinuousTrainRunner(base_dir, create_continuous_agent)
  else:
    raise ValueError('Unknown schedule: {}'.format(schedule))


@gin.configurable
class ContinuousRunner(base_run_experiment.Runner):
  """Object that handles running Dopamine experiments.

  This is mostly the same as discrete_domains.Runner, but is written solely for
  JAX/Flax agents.
  """

  def __init__(self,
               base_dir,
               create_agent_fn,
               create_environment_fn=gym_lib.create_gym_environment,
               checkpoint_file_prefix='ckpt',
               logging_file_prefix='log',
               log_every_n=1,
               num_iterations=200,
               training_steps=250000,
               evaluation_steps=125000,
               max_steps_per_episode=1000,
               clip_rewards=False,
               use_legacy_logger=True):
    """Initialize the Runner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as argument an environment, and
        returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
      checkpoint_file_prefix: str, the prefix to use for checkpoint files.
      logging_file_prefix: str, prefix to use for the log files.
      log_every_n: int, the frequency for writing logs.
      num_iterations: int, the iteration number threshold (must be greater than
        start_iteration).
      training_steps: int, the number of training steps to perform.
      evaluation_steps: int, the number of evaluation steps to perform.
      max_steps_per_episode: int, maximum number of steps after which an episode
        terminates.
      clip_rewards: bool, whether to clip rewards in [-1, 1].
      use_legacy_logger: bool, whether to use the legacy Logger. This will be
        deprecated soon, replaced with the new CollectorDispatcher setup.

    This constructor will take the following actions:
    - Initialize an environment.
    - Initialize a logger.
    - Initialize an agent.
    - Reload from the latest checkpoint, if available, and initialize the
      Checkpointer object.
    """
    assert base_dir is not None
    self._legacy_logger_enabled = use_legacy_logger
    self._logging_file_prefix = logging_file_prefix
    self._log_every_n = log_every_n
    self._num_iterations = num_iterations
    self._training_steps = training_steps
    self._evaluation_steps = evaluation_steps
    self._max_steps_per_episode = max_steps_per_episode
    self._base_dir = base_dir
    self._clip_rewards = clip_rewards
    self._create_directories()
    self._summary_writer = tensorboard.SummaryWriter(base_dir)
    self._environment = create_environment_fn()
    self._agent = create_agent_fn(self._environment,
                                  summary_writer=self._summary_writer)
    self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)

    # Create a collector dispatcher for metrics reporting.
    self._collector_dispatcher = collector_dispatcher.CollectorDispatcher(
        self._base_dir)
    set_collector_dispatcher_fn = getattr(
        self._agent, 'set_collector_dispatcher', None)
    if callable(set_collector_dispatcher_fn):
      set_collector_dispatcher_fn(self._collector_dispatcher)

  @property
  def _use_legacy_logger(self):
    if not hasattr(self, '_legacy_logger_enabled'):
      return True
    return self._legacy_logger_enabled

  @property
  def _has_collector_dispatcher(self):
    if not hasattr(self, '_collector_dispatcher'):
      return False
    return True

  @property
  def _fine_grained_print_to_console(self):
    if not hasattr(self, '_fine_grained_print_to_console_enabled'):
      return True
    return self._fine_grained_print_to_console_enabled

  def _save_tensorboard_summaries(self, iteration,
                                  num_episodes_train,
                                  average_reward_train,
                                  num_episodes_eval,
                                  average_reward_eval,
                                  average_steps_per_second):
    """Save statistics as tensorboard summaries.

    Args:
      iteration: int, The current iteration number.
      num_episodes_train: int, number of training episodes run.
      average_reward_train: float, The average training reward.
      num_episodes_eval: int, number of evaluation episodes run.
      average_reward_eval: float, The average evaluation reward.
      average_steps_per_second: float, The average number of steps per second.
    """
    metrics = [('Train/NumEpisodes', num_episodes_train),
               ('Train/AverageReturns', average_reward_train),
               ('Train/AverageStepsPerSecond', average_steps_per_second),
               ('Eval/NumEpisodes', num_episodes_eval),
               ('Eval/AverageReturns', average_reward_eval)]
    for name, value in metrics:
      self._summary_writer.scalar(name, value, iteration)
    self._summary_writer.flush()


@gin.configurable
class ContinuousTrainRunner(ContinuousRunner):
  """Object that handles running experiments.

  This is mostly the same as discrete_domains.TrainRunner, but is written solely
  for JAX/Flax agents.
  """

  def __init__(self, base_dir, create_agent_fn,
               create_environment_fn=gym_lib.create_gym_environment):
    """Initialize the TrainRunner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
    """
    logging.info('Creating ContinuousTrainRunner ...')
    super().__init__(base_dir, create_agent_fn, create_environment_fn)
    self._agent.eval_mode = False

  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction.

    An iteration involves running several episodes until a certain number of
    steps are obtained. This method differs from the `_run_one_iteration` method
    in the base `Runner` class in that it only runs the train phase.

    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.

    Returns:
      A dict containing summary statistics for this iteration.
    """
    statistics = iteration_statistics.IterationStatistics()
    num_episodes_train, average_reward_train, average_steps_per_second = (
        self._run_train_phase(statistics))

    if self._has_collector_dispatcher:
      self._collector_dispatcher.write([
          statistics_instance.StatisticsInstance('Train/NumEpisodes',
                                                 num_episodes_train,
                                                 iteration),
          statistics_instance.StatisticsInstance('Train/AverageReturns',
                                                 average_reward_train,
                                                 iteration),
          statistics_instance.StatisticsInstance('Train/AverageStepsPerSecond',
                                                 average_steps_per_second,
                                                 iteration),
      ])
    self._save_tensorboard_summaries(iteration, num_episodes_train,
                                     average_reward_train,
                                     average_steps_per_second)
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration, num_episodes,
                                  average_reward, average_steps_per_second):
    """Save statistics as tensorboard summaries."""
    metrics = [('Train/NumEpisodes', num_episodes),
               ('Train/AverageReturns', average_reward),
               ('Train/AverageStepsPerSecond', average_steps_per_second)]
    for name, value in metrics:
      self._summary_writer.scalar(name, value, iteration)
    self._summary_writer.flush()

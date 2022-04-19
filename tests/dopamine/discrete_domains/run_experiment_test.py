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
"""Tests for dopamine.common.run_experiment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil



from absl import flags
from dopamine.agents.dqn import dqn_agent
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import logger
from dopamine.discrete_domains import run_experiment
from dopamine.metrics import collector_dispatcher
from dopamine.metrics import statistics_instance
import gin.tf
import mock
import tensorflow as tf


FLAGS = flags.FLAGS


def _create_mock_checkpointer():
  mock_checkpointer = mock.Mock()
  test_dictionary = {'current_iteration': 1729,
                     'logs': 'logs'}
  mock_checkpointer.load_checkpoint.return_value = test_dictionary
  return mock_checkpointer


class MockEnvironment(object):
  """Mock environment for testing."""

  def __init__(self, max_steps=10):
    self._observation = 0
    self.max_steps = max_steps
    self.game_over = False

  def reset(self):
    self._observation = 0
    return self._observation

  def step(self, action):
    self._observation += 1
    reward_multiplier = -1 if action > 0 else 1
    reward = self._observation * reward_multiplier
    is_terminal = self._observation >= self.max_steps
    self.game_over = is_terminal

    unused = 0
    return (self._observation, reward, is_terminal, unused)

  def render(self, mode):
    pass


class MockLogger(object):
  """Class to mock the experiment logger."""

  def __init__(self, test_cls=None, run_asserts=True, data=None):
    self._test_cls = test_cls
    self._run_asserts = run_asserts
    self._iter = 0
    self._calls_to_set = 0
    self._calls_to_log = 0
    self.data = data

  def __setitem__(self, key, val):
    if self._run_asserts:
      self._test_cls.assertEqual('iteration_{:d}'.format(self._iter), key)
      self._test_cls.assertEqual('statistics', val)
      self._iter += 1
    self._calls_to_set += 1

  def log_to_file(self, filename_prefix, iteration_number):
    if self._run_asserts:
      self._test_cls.assertEqual(
          'prefix_{}'.format(self._iter - 1),
          '{}_{}'.format(filename_prefix, iteration_number))
    self._calls_to_log += 1


class RunExperimentTest(tf.test.TestCase):

  @mock.patch.object(gin, 'parse_config_files_and_bindings')
  def testLoadGinConfigs(self, mock_parse_config_files_and_bindings):
    gin_files = ['file1', 'file2', 'file3']
    gin_bindings = ['binding1', 'binding2']
    run_experiment.load_gin_configs(gin_files, gin_bindings)
    self.assertEqual(1, mock_parse_config_files_and_bindings.call_count)
    mock_args, mock_kwargs = mock_parse_config_files_and_bindings.call_args
    self.assertEqual(gin_files, mock_args[0])
    self.assertEqual(gin_bindings, mock_kwargs['bindings'])
    self.assertFalse(mock_kwargs['skip_unknown'])

  def testNoAgentName(self):
    with self.assertRaises(AssertionError):
      _ = run_experiment.create_agent(self.test_session(), mock.Mock())

  @mock.patch.object(dqn_agent, 'DQNAgent')
  def testCreateDQNAgent(self, mock_dqn_agent):
    def mock_fn(unused_sess, num_actions, summary_writer):
      del summary_writer
      return num_actions * 10

    mock_dqn_agent.side_effect = mock_fn
    environment = mock.Mock()
    environment.action_space.n = 7
    self.assertEqual(70, run_experiment.create_agent(self.test_session(),
                                                     environment,
                                                     agent_name='dqn'))

  @mock.patch.object(rainbow_agent, 'RainbowAgent')
  def testCreateRainbowAgent(self, mock_rainbow_agent):
    def mock_fn(unused_sess, num_actions, summary_writer):
      del summary_writer
      return num_actions * 10

    mock_rainbow_agent.side_effect = mock_fn
    environment = mock.Mock()
    environment.action_space.n = 7
    self.assertEqual(70, run_experiment.create_agent(self.test_session(),
                                                     environment,
                                                     agent_name='rainbow'))

  @mock.patch.object(implicit_quantile_agent, 'ImplicitQuantileAgent')
  def testCreateImplicitQuantileAgent(self, mock_implicit_quantile_agent):
    def mock_fn(unused_sess, num_actions, summary_writer):
      del summary_writer
      return num_actions * 10

    mock_implicit_quantile_agent.side_effect = mock_fn
    environment = mock.Mock()
    environment.action_space.n = 7
    self.assertEqual(70, run_experiment.create_agent(
        self.test_session(), environment, agent_name='implicit_quantile'))

  def testCreateRunnerUnknown(self):
    base_dir = '/tmp'
    with self.assertRaisesRegex(ValueError, 'Unknown schedule'):
      run_experiment.create_runner(base_dir,
                                   'Unknown schedule')

  @mock.patch.object(run_experiment, 'Runner')
  @mock.patch.object(run_experiment, 'create_agent')
  def testCreateRunner(self, mock_create_agent, mock_runner_constructor):
    base_dir = '/tmp'
    run_experiment.create_runner(base_dir)
    self.assertEqual(1, mock_runner_constructor.call_count)
    mock_args, _ = mock_runner_constructor.call_args
    self.assertEqual(base_dir, mock_args[0])
    self.assertEqual(mock_create_agent, mock_args[1])

  @mock.patch.object(run_experiment, 'TrainRunner')
  @mock.patch.object(run_experiment, 'create_agent')
  def testCreateTrainRunner(self, mock_create_agent, mock_runner_constructor):
    base_dir = '/tmp'
    run_experiment.create_runner(base_dir,
                                 schedule='continuous_train')
    self.assertEqual(1, mock_runner_constructor.call_count)
    mock_args, _ = mock_runner_constructor.call_args
    self.assertEqual(base_dir, mock_args[0])
    self.assertEqual(mock_create_agent, mock_args[1])


class RunnerTest(tf.test.TestCase):

  def _agent_step(self, reward, observation):
    # We verify that rewards are clipped (and set by MockEnvironment as a
    # function of observation)
    expected_reward = 1 if observation % 2 else -1
    self.assertEqual(expected_reward, reward)
    return observation % 2

  def setUp(self):
    super(RunnerTest, self).setUp()
    self._agent = mock.Mock()
    self._agent.begin_episode.side_effect = lambda x: 0
    self._agent.step.side_effect = self._agent_step
    def create_agent_fn(unused_x, unused_y, summary_writer):
      assert isinstance(summary_writer, str)
      self._agent.summary_writer = tf.compat.v1.summary.FileWriter(
          summary_writer)
      config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
      config.gpu_options.allow_growth = True
      self._agent._sess = tf.compat.v1.Session('', config=config)
      return self._agent

    self._create_agent_fn = create_agent_fn
    self._test_subdir = '/tmp/dopamine_tests'
    shutil.rmtree(self._test_subdir, ignore_errors=True)
    os.makedirs(self._test_subdir)

  @mock.patch.object(checkpointer, 'get_latest_checkpoint_number')
  def testInitializeCheckpointingWithNoCheckpointFile(self, mock_get_latest):
    mock_get_latest.return_value = -1
    base_dir = '/does/not/exist'
    with self.assertRaisesRegex(tf.errors.PermissionDeniedError,
                                '.*/does.*'):
      run_experiment.Runner(base_dir, self._create_agent_fn, mock.Mock)

  @mock.patch.object(checkpointer, 'get_latest_checkpoint_number')
  @mock.patch.object(checkpointer, 'Checkpointer')
  @mock.patch.object(logger, 'Logger')
  def testInitializeCheckpointingWhenCheckpointUnbundleFails(
      self, mock_logger_constructor, mock_checkpointer_constructor,
      mock_get_latest):
    mock_checkpointer = _create_mock_checkpointer()
    mock_checkpointer_constructor.return_value = mock_checkpointer
    latest_checkpoint = 7
    mock_get_latest.return_value = latest_checkpoint
    agent = mock.Mock()
    agent.unbundle.return_value = False
    mock_logger = mock.Mock()
    mock_logger_constructor.return_value = mock_logger
    runner = run_experiment.Runner(self._test_subdir,
                                   lambda x, y, summary_writer: agent,
                                   mock.Mock)
    self.assertEqual(0, runner._start_iteration)
    self.assertEqual(1, mock_checkpointer.load_checkpoint.call_count)
    self.assertEqual(1, agent.unbundle.call_count)
    mock_args, _ = agent.unbundle.call_args
    self.assertEqual('{}/checkpoints'.format(self._test_subdir), mock_args[0])
    self.assertEqual(latest_checkpoint, mock_args[1])
    expected_dictionary = {'current_iteration': 1729,
                           'logs': 'logs'}
    self.assertDictEqual(expected_dictionary, mock_args[2])

  @mock.patch.object(checkpointer, 'get_latest_checkpoint_number')
  def testInitializeCheckpointingWhenCheckpointUnbundleSucceeds(
      self, mock_get_latest):
    latest_checkpoint = 7
    mock_get_latest.return_value = latest_checkpoint
    logs_data = {'a': 1, 'b': 2}
    current_iteration = 1729
    checkpoint_data = {'current_iteration': current_iteration,
                       'logs': logs_data}
    checkpoint_dir = os.path.join(self._test_subdir, 'checkpoints')
    checkpoint = checkpointer.Checkpointer(checkpoint_dir, 'ckpt')
    checkpoint.save_checkpoint(latest_checkpoint, checkpoint_data)
    mock_agent = mock.Mock()
    mock_agent.unbundle.return_value = True
    runner = run_experiment.Runner(self._test_subdir,
                                   lambda x, y, summary_writer: mock_agent,
                                   mock.Mock)
    expected_iteration = current_iteration + 1
    self.assertEqual(expected_iteration, runner._start_iteration)
    self.assertDictEqual(logs_data, runner._logger.data)
    mock_agent.unbundle.assert_called_once_with(
        checkpoint_dir, latest_checkpoint, checkpoint_data)

  def testRunOneEpisode(self):
    max_steps_per_episode = 11
    environment = MockEnvironment()
    runner = run_experiment.Runner(
        self._test_subdir, self._create_agent_fn, lambda: environment,
        max_steps_per_episode=max_steps_per_episode)
    step_number, total_reward = runner._run_one_episode()
    self.assertEqual(self._agent.step.call_count, environment.max_steps - 1)
    self.assertEqual(self._agent.end_episode.call_count, 1)
    self.assertEqual(environment.max_steps, step_number)
    # Expected reward will be \sum_{i=0}^{9} (-1)**i * i = -5
    self.assertEqual(-5, total_reward)

  def testRunOneEpisodeWithLowMaxSteps(self):
    max_steps_per_episode = 2
    environment = MockEnvironment()
    runner = run_experiment.Runner(
        self._test_subdir, self._create_agent_fn, lambda: environment,
        max_steps_per_episode=max_steps_per_episode)
    step_number, total_reward = runner._run_one_episode()
    self.assertEqual(self._agent.step.call_count, max_steps_per_episode - 1)
    self.assertEqual(self._agent.end_episode.call_count, 1)
    self.assertEqual(max_steps_per_episode, step_number)
    self.assertEqual(-1, total_reward)

  def testRunOnePhase(self):
    max_steps = 10
    environment_steps = 2
    environment = MockEnvironment(max_steps=environment_steps)
    statistics = []
    runner = run_experiment.Runner(
        self._test_subdir, self._create_agent_fn, lambda: environment)
    step_number, sum_returns, num_episodes = runner._run_one_phase(
        max_steps, statistics, 'test')
    calls_to_run_episode = int(max_steps / environment_steps)
    self.assertEqual(self._agent.step.call_count, calls_to_run_episode)
    self.assertEqual(self._agent.end_episode.call_count, calls_to_run_episode)
    self.assertEqual(max_steps, step_number)
    self.assertEqual(-1 * calls_to_run_episode, sum_returns)
    self.assertEqual(calls_to_run_episode, num_episodes)
    expected_statistics = []
    for _ in range(calls_to_run_episode):
      expected_statistics.append({
          'test_episode_lengths': 2,
          'test_episode_returns': -1
      })
    self.assertEqual(len(expected_statistics), len(statistics))
    for i in range(len(statistics)):
      self.assertDictEqual(expected_statistics[i], statistics[i])

  @mock.patch.object(collector_dispatcher, 'CollectorDispatcher')
  def testRunOneIteration(self, mock_collector_dispatcher):
    mock_collector_dispatcher.return_value = mock.Mock()
    environment_steps = 2
    environment = MockEnvironment(max_steps=environment_steps)
    training_steps = 20
    evaluation_steps = 10
    runner = run_experiment.Runner(
        self._test_subdir, self._create_agent_fn, lambda: environment,
        training_steps=training_steps,
        evaluation_steps=evaluation_steps)
    dictionary = runner._run_one_iteration(1)
    train_calls = int(training_steps / environment_steps)
    eval_calls = int(evaluation_steps / environment_steps)
    expected_dictionary = {
        'train_episode_lengths': [2 for _ in range(train_calls)],
        'train_episode_returns': [-1 for _ in range(train_calls)],
        'train_average_return': [-1],
        'eval_episode_lengths': [2 for _ in range(eval_calls)],
        'eval_episode_returns': [-1 for _ in range(eval_calls)],
        'eval_average_return': [-1]
    }
    for k in expected_dictionary:
      self.assertEqual(expected_dictionary[k], dictionary[k])
    # Also verify that average number of steps per second is present and
    # positive.
    self.assertLen(dictionary['train_average_steps_per_second'], 1)
    self.assertGreater(dictionary['train_average_steps_per_second'][0], 0)
    self.assertEqual(1, runner._collector_dispatcher.write.call_count)
    keys = ['Train/NumEpisodes', 'Train/AverageReturns',
            'Train/AverageStepsPerSecond', 'Eval/NumEpisodes',
            'Eval/AverageReturns']
    vals = [train_calls, -1.0, None, eval_calls, -1.0]
    arg_pos = 0
    for key, val in zip(keys, vals):
      if val is None:
        arg_pos += 1
        continue  # Ignore steps per second
      self.assertEqual(
          statistics_instance.StatisticsInstance(key, val, 1),
          runner._collector_dispatcher.write.call_args_list[0][0][0][arg_pos])
      arg_pos += 1

  @mock.patch.object(logger, 'Logger')
  def testLogExperiment(self, mock_logger_constructor):
    log_every_n = 2
    logging_file_prefix = 'prefix'
    statistics = 'statistics'
    experiment_logger = MockLogger(test_cls=self)
    mock_logger_constructor.return_value = experiment_logger
    runner = run_experiment.Runner(
        self._test_subdir, self._create_agent_fn, mock.Mock,
        logging_file_prefix=logging_file_prefix,
        log_every_n=log_every_n)
    num_iterations = 10
    for i in range(num_iterations):
      runner._log_experiment(i, statistics)
    self.assertEqual(num_iterations, experiment_logger._calls_to_set)
    self.assertEqual((num_iterations / log_every_n),
                     experiment_logger._calls_to_log)

  @mock.patch.object(logger, 'Logger')
  def testLogExperimentWithoutLegacyLogging(self, mock_logger_constructor):
    log_every_n = 2
    logging_file_prefix = 'prefix'
    statistics = 'statistics'
    experiment_logger = MockLogger(test_cls=self)
    mock_logger_constructor.return_value = experiment_logger
    runner = run_experiment.Runner(
        self._test_subdir, self._create_agent_fn, mock.Mock,
        logging_file_prefix=logging_file_prefix,
        log_every_n=log_every_n,
        use_legacy_logger=False)
    num_iterations = 10
    for i in range(num_iterations):
      # This should not do anything.
      runner._log_experiment(i, statistics)
    # With legacy logging turned off, it should never call the Logger.
    self.assertEqual(0, experiment_logger._calls_to_set)
    self.assertEqual(0, experiment_logger._calls_to_log)

  @mock.patch.object(checkpointer, 'Checkpointer')
  @mock.patch.object(logger, 'Logger')
  def testCheckpointExperiment(self, mock_logger_constructor,
                               mock_checkpointer_constructor):
    checkpoint_dir = os.path.join(self._test_subdir, 'checkpoints')
    test_dict = {'test': 1}
    iteration = 1729

    def bundle_and_checkpoint(x, y):
      self.assertEqual(checkpoint_dir, x)
      self.assertEqual(iteration, y)
      return test_dict

    self._agent.bundle_and_checkpoint.side_effect = bundle_and_checkpoint
    experiment_checkpointer = mock.Mock()
    mock_checkpointer_constructor.return_value = experiment_checkpointer
    logs_data = {'one': 1, 'two': 2}
    mock_logger = MockLogger(run_asserts=False, data=logs_data)
    mock_logger_constructor.return_value = mock_logger
    runner = run_experiment.Runner(
        self._test_subdir, self._create_agent_fn, mock.Mock)
    runner._checkpoint_experiment(iteration)
    self.assertEqual(1, experiment_checkpointer.save_checkpoint.call_count)
    mock_args, _ = experiment_checkpointer.save_checkpoint.call_args
    self.assertEqual(iteration, mock_args[0])
    test_dict['logs'] = logs_data
    test_dict['current_iteration'] = iteration
    self.assertDictEqual(test_dict, mock_args[1])

  @mock.patch.object(checkpointer, 'Checkpointer')
  @mock.patch.object(logger, 'Logger')
  def testRunExperimentWithInconsistentRange(self, mock_logger_constructor,
                                             mock_checkpointer_constructor):
    experiment_logger = MockLogger()
    mock_logger_constructor.return_value = experiment_logger
    experiment_checkpointer = mock.Mock()
    mock_checkpointer_constructor.return_value = experiment_checkpointer
    runner = run_experiment.Runner(
        self._test_subdir, self._create_agent_fn, mock.Mock,
        num_iterations=0)
    runner.run_experiment()
    self.assertEqual(0, experiment_checkpointer.save_checkpoint.call_count)
    self.assertEqual(0, experiment_logger._calls_to_set)
    self.assertEqual(0, experiment_logger._calls_to_log)

  @mock.patch.object(checkpointer, 'get_latest_checkpoint_number')
  @mock.patch.object(checkpointer, 'Checkpointer')
  @mock.patch.object(logger, 'Logger')
  def testRunExperiment(self, mock_logger_constructor,
                        mock_checkpointer_constructor,
                        mock_get_latest):
    log_every_n = 1
    environment = MockEnvironment()
    experiment_logger = MockLogger(run_asserts=False)
    mock_logger_constructor.return_value = experiment_logger
    experiment_checkpointer = mock.Mock()
    start_iteration = 1729
    mock_get_latest.return_value = start_iteration
    def load_checkpoint(_):
      return {'logs': 'log_data', 'current_iteration': start_iteration - 1}

    experiment_checkpointer.load_checkpoint.side_effect = load_checkpoint
    mock_checkpointer_constructor.return_value = experiment_checkpointer
    def bundle_and_checkpoint(x, y):
      del x, y  # Unused.
      return {'test': 1}

    self._agent.bundle_and_checkpoint.side_effect = bundle_and_checkpoint
    num_iterations = 10
    self._agent.unbundle.return_value = True
    end_iteration = start_iteration + num_iterations
    runner = run_experiment.Runner(
        self._test_subdir, self._create_agent_fn, lambda: environment,
        log_every_n=log_every_n,
        num_iterations=end_iteration,
        training_steps=1,
        evaluation_steps=1)
    self.assertEqual(start_iteration, runner._start_iteration)
    runner.run_experiment()
    self.assertEqual(num_iterations,
                     experiment_checkpointer.save_checkpoint.call_count)
    self.assertEqual(num_iterations, experiment_logger._calls_to_set)
    self.assertEqual(num_iterations, experiment_logger._calls_to_log)
    glob_string = '{}/events.out.tfevents.*'.format(self._test_subdir)
    self.assertNotEmpty(tf.io.gfile.glob(glob_string))

  @mock.patch.object(collector_dispatcher, 'CollectorDispatcher')
  def testCollectorDispatcherSetup(self, mock_collector_dispatcher):
    environment = MockEnvironment()
    mock_collector_dispatcher.return_value = 'CD'
    self._agent.set_collector_dispatcher = mock.Mock()
    _ = run_experiment.Runner(
        self._test_subdir, self._create_agent_fn, lambda: environment)
    self.assertEqual(1, mock_collector_dispatcher.call_count)
    self.assertEqual(1, self._agent.set_collector_dispatcher.call_count)
    self.assertEqual(
        'CD', self._agent.set_collector_dispatcher.call_args_list[0][0][0])


if __name__ == '__main__':
  tf.compat.v1.disable_v2_behavior()
  tf.test.main()

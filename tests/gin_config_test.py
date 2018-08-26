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
"""Tests for establishing correct dependency injection with gin."""

import datetime
import os
import shutil



from absl import flags
from dopamine.atari import run_experiment
from dopamine.atari import train
import tensorflow as tf

import gin.tf

FLAGS = flags.FLAGS


class GinConfigTest(tf.test.TestCase):
  """Tests for configuring Atari agents using gin.

  """

  def setUp(self):
    FLAGS.base_dir = os.path.join(
        '/tmp/dopamine_tests',
        datetime.datetime.utcnow().strftime('run_%Y_%m_%d_%H_%M_%S'))
    self._checkpoint_dir = os.path.join(FLAGS.base_dir, 'checkpoints')
    self._logging_dir = os.path.join(FLAGS.base_dir, 'logs')
    self._videos_dir = os.path.join(FLAGS.base_dir, 'videos')
    gin.clear_config()

  def testDefaultGinDqn(self):
    """Test DQNAgent configuration using the default gin config."""
    tf.logging.info('####### Training the DQN agent #####')
    tf.logging.info('####### DQN base_dir: {}'.format(FLAGS.base_dir))
    FLAGS.agent_name = 'dqn'
    FLAGS.gin_files = ['dopamine/agents/dqn/configs/dqn.gin']
    FLAGS.gin_bindings = [
        'WrappedReplayBuffer.replay_capacity = 100'  # To prevent OOM.
    ]
    run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    runner = run_experiment.Runner(FLAGS.base_dir, train.create_agent)
    self.assertIsInstance(runner._agent.optimizer, tf.train.RMSPropOptimizer)
    self.assertNear(0.00025, runner._agent.optimizer._learning_rate, 0.0001)
    shutil.rmtree(FLAGS.base_dir)

  def testOverrideRunnerParams(self):
    """Test DQNAgent configuration using the default gin config."""
    tf.logging.info('####### Training the DQN agent #####')
    tf.logging.info('####### DQN base_dir: {}'.format(FLAGS.base_dir))
    FLAGS.agent_name = 'dqn'
    FLAGS.gin_files = ['dopamine/agents/dqn/configs/dqn.gin']
    FLAGS.gin_bindings = [
        'TrainRunner.base_dir = "{}"'.format(FLAGS.base_dir),
        'Runner.log_every_n = 1729',
        'WrappedReplayBuffer.replay_capacity = 100'  # To prevent OOM.
    ]
    run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    runner = run_experiment.TrainRunner(create_agent_fn=train.create_agent)
    self.assertEqual(runner._base_dir, FLAGS.base_dir)
    self.assertEqual(runner._log_every_n, 1729)
    shutil.rmtree(FLAGS.base_dir)

  def testDefaultGinRmspropDqn(self):
    """Test DQNAgent configuration overridden with RMSPropOptimizer."""
    tf.logging.info('####### Training the DQN agent #####')
    tf.logging.info('####### DQN base_dir: {}'.format(FLAGS.base_dir))
    FLAGS.agent_name = 'dqn'
    FLAGS.gin_files = ['dopamine/agents/dqn/configs/dqn.gin']
    FLAGS.gin_bindings = [
        'DQNAgent.optimizer = @tf.train.RMSPropOptimizer()',
        'tf.train.RMSPropOptimizer.learning_rate = 100',
        'WrappedReplayBuffer.replay_capacity = 100'  # To prevent OOM.
    ]
    run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    runner = run_experiment.Runner(FLAGS.base_dir, train.create_agent)
    self.assertIsInstance(runner._agent.optimizer, tf.train.RMSPropOptimizer)
    self.assertEqual(100, runner._agent.optimizer._learning_rate)
    shutil.rmtree(FLAGS.base_dir)

  def testOverrideGinDqn(self):
    """Test DQNAgent configuration overridden with AdamOptimizer."""
    tf.logging.info('####### Training the DQN agent #####')
    tf.logging.info('####### DQN base_dir: {}'.format(FLAGS.base_dir))
    FLAGS.agent_name = 'dqn'
    FLAGS.gin_files = ['dopamine/agents/dqn/configs/dqn.gin']
    FLAGS.gin_bindings = [
        'DQNAgent.optimizer = @tf.train.AdamOptimizer()',
        'tf.train.AdamOptimizer.learning_rate = 100',
        'WrappedReplayBuffer.replay_capacity = 100'  # To prevent OOM.
    ]

    run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    runner = run_experiment.Runner(FLAGS.base_dir, train.create_agent)
    self.assertIsInstance(runner._agent.optimizer, tf.train.AdamOptimizer)
    self.assertEqual(100, runner._agent.optimizer._lr)
    shutil.rmtree(FLAGS.base_dir)

  def testDefaultGinRainbow(self):
    """Test RainbowAgent default configuration using default gin."""
    tf.logging.info('####### Training the RAINBOW agent #####')
    tf.logging.info('####### RAINBOW base_dir: {}'.format(FLAGS.base_dir))
    FLAGS.agent_name = 'rainbow'
    FLAGS.gin_files = [
        'dopamine/agents/rainbow/configs/rainbow.gin'
    ]
    FLAGS.gin_bindings = [
        'WrappedReplayBuffer.replay_capacity = 100'  # To prevent OOM.
    ]
    run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    runner = run_experiment.Runner(FLAGS.base_dir, train.create_agent)
    self.assertIsInstance(runner._agent.optimizer, tf.train.AdamOptimizer)
    self.assertNear(0.0000625, runner._agent.optimizer._lr, 0.0001)
    shutil.rmtree(FLAGS.base_dir)

  def testOverrideGinRainbow(self):
    """Test RainbowAgent configuration overridden with RMSPropOptimizer."""
    tf.logging.info('####### Training the RAINBOW agent #####')
    tf.logging.info('####### RAINBOW base_dir: {}'.format(FLAGS.base_dir))
    FLAGS.agent_name = 'rainbow'
    FLAGS.gin_files = [
        'dopamine/agents/rainbow/configs/rainbow.gin',
    ]
    FLAGS.gin_bindings = [
        'RainbowAgent.optimizer = @tf.train.RMSPropOptimizer()',
        'tf.train.RMSPropOptimizer.learning_rate = 100',
        'WrappedReplayBuffer.replay_capacity = 100'  # To prevent OOM.
    ]
    run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    runner = run_experiment.Runner(FLAGS.base_dir, train.create_agent)
    self.assertIsInstance(runner._agent.optimizer, tf.train.RMSPropOptimizer)
    self.assertEqual(100, runner._agent.optimizer._learning_rate)
    shutil.rmtree(FLAGS.base_dir)

  def testDefaultDQNConfig(self):
    """Verify the default DQN configuration."""
    FLAGS.agent_name = 'dqn'
    run_experiment.load_gin_configs(
        ['dopamine/agents/dqn/configs/dqn.gin'], [])
    agent = train.create_agent(self.test_session(),
                               run_experiment.create_atari_environment('Pong'))
    self.assertEqual(agent.gamma, 0.99)
    self.assertEqual(agent.update_horizon, 1)
    self.assertEqual(agent.min_replay_history, 20000)
    self.assertEqual(agent.update_period, 4)
    self.assertEqual(agent.target_update_period, 8000)
    self.assertEqual(agent.epsilon_train, 0.01)
    self.assertEqual(agent.epsilon_eval, 0.001)
    self.assertEqual(agent.epsilon_decay_period, 250000)
    self.assertEqual(agent._replay.memory._replay_capacity, 1000000)
    self.assertEqual(agent._replay.memory._batch_size, 32)

  def testDefaultC51Config(self):
    """Verify the default C51 configuration."""
    FLAGS.agent_name = 'rainbow'
    run_experiment.load_gin_configs(
        ['dopamine/agents/rainbow/configs/c51.gin'], [])
    agent = train.create_agent(self.test_session(),
                               run_experiment.create_atari_environment('Pong'))
    self.assertEqual(agent._num_atoms, 51)
    support = self.evaluate(agent._support)
    self.assertEqual(min(support), -10.)
    self.assertEqual(max(support), 10.)
    self.assertEqual(len(support), 51)
    self.assertEqual(agent.gamma, 0.99)
    self.assertEqual(agent.update_horizon, 1)
    self.assertEqual(agent.min_replay_history, 20000)
    self.assertEqual(agent.update_period, 4)
    self.assertEqual(agent.target_update_period, 8000)
    self.assertEqual(agent.epsilon_train, 0.01)
    self.assertEqual(agent.epsilon_eval, 0.001)
    self.assertEqual(agent.epsilon_decay_period, 250000)
    self.assertEqual(agent._replay.memory._replay_capacity, 1000000)
    self.assertEqual(agent._replay.memory._batch_size, 32)

  def testDefaultRainbowConfig(self):
    """Verify the default Rainbow configuration."""
    FLAGS.agent_name = 'rainbow'
    run_experiment.load_gin_configs(
        ['dopamine/agents/rainbow/configs/rainbow.gin'], [])
    agent = train.create_agent(self.test_session(),
                               run_experiment.create_atari_environment('Pong'))
    self.assertEqual(agent._num_atoms, 51)
    support = self.evaluate(agent._support)
    self.assertEqual(min(support), -10.)
    self.assertEqual(max(support), 10.)
    self.assertEqual(len(support), 51)
    self.assertEqual(agent.gamma, 0.99)
    self.assertEqual(agent.update_horizon, 3)
    self.assertEqual(agent.min_replay_history, 20000)
    self.assertEqual(agent.update_period, 4)
    self.assertEqual(agent.target_update_period, 8000)
    self.assertEqual(agent.epsilon_train, 0.01)
    self.assertEqual(agent.epsilon_eval, 0.001)
    self.assertEqual(agent.epsilon_decay_period, 250000)
    self.assertEqual(agent._replay.memory._replay_capacity, 1000000)
    self.assertEqual(agent._replay.memory._batch_size, 32)

  def testDefaultGinImplicitQuantileIcml(self):
    """Test default ImplicitQuantile configuration using ICML gin."""
    tf.logging.info('###### Training the Implicit Quantile agent #####')
    FLAGS.agent_name = 'implicit_quantile'
    FLAGS.base_dir = os.path.join(
        '/tmp/dopamine_tests',
        datetime.datetime.utcnow().strftime('run_%Y_%m_%d_%H_%M_%S'))
    tf.logging.info('###### IQN base dir: {}'.format(FLAGS.base_dir))
    FLAGS.gin_files = ['dopamine/agents/'
                       'implicit_quantile/configs/implicit_quantile_icml.gin']
    FLAGS.gin_bindings = [
        'Runner.num_iterations=0',
    ]
    run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    runner = run_experiment.Runner(FLAGS.base_dir, train.create_agent)
    self.assertEqual(1000000, runner._agent._replay.memory._replay_capacity)
    shutil.rmtree(FLAGS.base_dir)

  def testOverrideGinImplicitQuantileIcml(self):
    """Test ImplicitQuantile configuration overriding using ICML gin."""
    tf.logging.info('###### Training the Implicit Quantile agent #####')
    FLAGS.agent_name = 'implicit_quantile'
    FLAGS.base_dir = os.path.join(
        '/tmp/dopamine_tests',
        datetime.datetime.utcnow().strftime('run_%Y_%m_%d_%H_%M_%S'))
    tf.logging.info('###### IQN base dir: {}'.format(FLAGS.base_dir))
    FLAGS.gin_files = ['dopamine/agents/'
                       'implicit_quantile/configs/implicit_quantile_icml.gin']
    FLAGS.gin_bindings = [
        'Runner.num_iterations=0',
        'WrappedPrioritizedReplayBuffer.replay_capacity = 1000',
    ]
    run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    runner = run_experiment.Runner(FLAGS.base_dir, train.create_agent)
    self.assertEqual(1000, runner._agent._replay.memory._replay_capacity)
    shutil.rmtree(FLAGS.base_dir)

  def testOverrideGinImplicitQuantile(self):
    """Test ImplicitQuantile configuration overriding using IQN gin."""
    tf.logging.info('###### Training the Implicit Quantile agent #####')
    FLAGS.agent_name = 'implicit_quantile'
    FLAGS.base_dir = os.path.join(
        '/tmp/dopamine_tests',
        datetime.datetime.utcnow().strftime('run_%Y_%m_%d_%H_%M_%S'))
    tf.logging.info('###### IQN base dir: {}'.format(FLAGS.base_dir))
    FLAGS.gin_files = ['dopamine/agents/'
                       'implicit_quantile/configs/implicit_quantile.gin']
    FLAGS.gin_bindings = [
        'Runner.num_iterations=0',
        'WrappedPrioritizedReplayBuffer.replay_capacity = 1000',
    ]
    run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    runner = run_experiment.Runner(FLAGS.base_dir, train.create_agent)
    self.assertEqual(1000, runner._agent._replay.memory._replay_capacity)
    shutil.rmtree(FLAGS.base_dir)


if __name__ == '__main__':
  tf.test.main()

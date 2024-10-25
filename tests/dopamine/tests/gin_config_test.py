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
"""Tests for establishing correct dependency injection with gin."""

import datetime
import os
import shutil

from absl import flags
from absl import logging
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import run_experiment
import gin
import optax
import tensorflow as tf


FLAGS = flags.FLAGS


class GinConfigTest(tf.test.TestCase):
  """Tests for configuring Atari agents using gin.

  """

  def setUp(self):
    super(GinConfigTest, self).setUp()
    self._base_dir = os.path.join(
        '/tmp/dopamine_tests',
        datetime.datetime.utcnow().strftime('run_%Y_%m_%d_%H_%M_%S'),
    )
    self._checkpoint_dir = os.path.join(self._base_dir, 'checkpoints')
    self._logging_dir = os.path.join(self._base_dir, 'logs')
    self._videos_dir = os.path.join(self._base_dir, 'videos')
    gin.clear_config()

  def testDefaultGinDqn(self):
    """Test DQNAgent configuration using the default gin config."""
    logging.info('####### Training the DQN agent #####')
    logging.info('####### DQN base_dir: %s', self._base_dir)
    gin_files = ['dopamine/jax/agents/dqn/configs/dqn.gin']
    gin_bindings = [
        'ReplayBuffer.max_capacity = 100',  # To prevent OOM.
        "create_agent.agent_name = 'jax_dqn'",
    ]
    run_experiment.load_gin_configs(gin_files, gin_bindings)
    runner = run_experiment.Runner(
        self._base_dir,
        run_experiment.create_agent,
        atari_lib.create_atari_environment,
    )
    self.assertIsInstance(
        runner._agent.optimizer, optax._src.base.GradientTransformationExtraArgs
    )
    shutil.rmtree(self._base_dir)

  def testOverrideRunnerParams(self):
    """Test DQNAgent configuration using the default gin config."""
    logging.info('####### Training the DQN agent #####')
    logging.info('####### DQN base_dir: %s', self._base_dir)
    gin_files = ['dopamine/jax/agents/dqn/configs/dqn.gin']
    gin_bindings = [
        'TrainRunner.base_dir = "{}"'.format(self._base_dir),
        'Runner.log_every_n = 1729',
        'ReplayBuffer.max_capacity = 100',  # To prevent OOM.
        "create_agent.agent_name = 'jax_dqn'",
    ]
    run_experiment.load_gin_configs(gin_files, gin_bindings)
    runner = run_experiment.TrainRunner(
        create_agent_fn=run_experiment.create_agent,
        create_environment_fn=atari_lib.create_atari_environment,
    )
    self.assertEqual(runner._base_dir, self._base_dir)
    self.assertEqual(runner._log_every_n, 1729)
    shutil.rmtree(self._base_dir)

  def testDefaultGinRmspropDqn(self):
    """Test DQNAgent configuration overridden with RMSPropOptimizer."""
    logging.info('####### Training the DQN agent #####')
    logging.info('####### DQN base_dir: %s', self._base_dir)
    gin_files = ['dopamine/jax/agents/dqn/configs/dqn.gin']
    gin_bindings = [
        'ReplayBuffer.max_capacity = 100',  # To prevent OOM.
        "create_agent.agent_name = 'jax_dqn'",
    ]
    run_experiment.load_gin_configs(gin_files, gin_bindings)
    runner = run_experiment.Runner(
        self._base_dir,
        run_experiment.create_agent,
        atari_lib.create_atari_environment,
    )
    self.assertIsInstance(
        runner._agent.optimizer, optax._src.base.GradientTransformationExtraArgs
    )
    shutil.rmtree(self._base_dir)

  def testOverrideGinDqn(self):
    """Test DQNAgent configuration overridden with AdamOptimizer."""
    logging.info('####### Training the DQN agent #####')
    logging.info('####### DQN base_dir: %s', self._base_dir)
    gin_files = ['dopamine/jax/agents/dqn/configs/dqn.gin']
    gin_bindings = [
        'ReplayBuffer.max_capacity = 100',  # To prevent OOM.
        "create_agent.agent_name = 'jax_dqn'",
    ]

    run_experiment.load_gin_configs(gin_files, gin_bindings)
    runner = run_experiment.Runner(
        self._base_dir,
        run_experiment.create_agent,
        atari_lib.create_atari_environment,
    )
    self.assertIsInstance(
        runner._agent.optimizer, optax._src.base.GradientTransformationExtraArgs
    )
    shutil.rmtree(self._base_dir)

  def testDefaultGinRainbow(self):
    """Test RainbowAgent default configuration using default gin."""
    logging.info('####### Training the RAINBOW agent #####')
    logging.info('####### RAINBOW base_dir: %s', self._base_dir)
    gin_files = [
        'dopamine/jax/agents/rainbow/configs/rainbow.gin'
    ]
    gin_bindings = [
        'ReplayBuffer.max_capacity = 100',  # To prevent OOM.
        "create_agent.agent_name = 'jax_rainbow'",
    ]
    run_experiment.load_gin_configs(gin_files, gin_bindings)
    runner = run_experiment.Runner(
        self._base_dir,
        run_experiment.create_agent,
        atari_lib.create_atari_environment,
    )
    self.assertIsInstance(
        runner._agent.optimizer, optax._src.base.GradientTransformationExtraArgs
    )
    shutil.rmtree(self._base_dir)

  def testOverrideGinRainbow(self):
    """Test RainbowAgent configuration overridden with RMSPropOptimizer."""
    logging.info('####### Training the RAINBOW agent #####')
    logging.info('####### RAINBOW base_dir: %s', self._base_dir)
    gin_files = [
        'dopamine/jax/agents/rainbow/configs/rainbow.gin',
    ]
    gin_bindings = [
        'ReplayBuffer.max_capacity = 100',  # To prevent OOM.
        "create_agent.agent_name = 'jax_rainbow'",
    ]
    run_experiment.load_gin_configs(gin_files, gin_bindings)
    runner = run_experiment.Runner(
        self._base_dir,
        run_experiment.create_agent,
        atari_lib.create_atari_environment,
    )
    self.assertIsInstance(
        runner._agent.optimizer, optax._src.base.GradientTransformationExtraArgs
    )
    shutil.rmtree(self._base_dir)

  def testDefaultDQNConfig(self):
    """Verify the default DQN configuration."""
    run_experiment.load_gin_configs(
        ['dopamine/jax/agents/dqn/configs/dqn.gin'], []
    )
    agent = run_experiment.create_agent(
        None, atari_lib.create_atari_environment(game_name='Pong')
    )
    self.assertEqual(agent.gamma, 0.99)
    self.assertEqual(agent.update_horizon, 1)
    self.assertEqual(agent.min_replay_history, 20000)
    self.assertEqual(agent.update_period, 4)
    self.assertEqual(agent.target_update_period, 8000)
    self.assertEqual(agent.epsilon_train, 0.01)
    self.assertEqual(agent.epsilon_eval, 0.001)
    self.assertEqual(agent.epsilon_decay_period, 250000)
    self.assertEqual(agent._replay._max_capacity, 1000000)
    self.assertEqual(agent._replay._batch_size, 32)

  def testDefaultC51Config(self):
    """Verify the default C51 configuration."""
    run_experiment.load_gin_configs(
        ['dopamine/jax/agents/rainbow/configs/c51.gin'], []
    )
    agent = run_experiment.create_agent(
        None, atari_lib.create_atari_environment(game_name='Pong')
    )
    self.assertEqual(agent._num_atoms, 51)
    support = agent._support
    self.assertEqual(min(support), -10.0)
    self.assertEqual(max(support), 10.0)
    self.assertLen(support, 51)
    self.assertEqual(agent.gamma, 0.99)
    self.assertEqual(agent.update_horizon, 1)
    self.assertEqual(agent.min_replay_history, 20000)
    self.assertEqual(agent.update_period, 4)
    self.assertEqual(agent.target_update_period, 8000)
    self.assertEqual(agent.epsilon_train, 0.01)
    self.assertEqual(agent.epsilon_eval, 0.001)
    self.assertEqual(agent.epsilon_decay_period, 250000)
    self.assertEqual(agent._replay._max_capacity, 1000000)
    self.assertEqual(agent._replay._batch_size, 32)

  def testDefaultRainbowConfig(self):
    """Verify the default Rainbow configuration."""
    run_experiment.load_gin_configs(
        ['dopamine/jax/agents/rainbow/configs/rainbow.gin'], []
    )
    agent = run_experiment.create_agent(
        None, atari_lib.create_atari_environment(game_name='Pong')
    )
    self.assertEqual(agent._num_atoms, 51)
    support = agent._support
    self.assertEqual(min(support), -10.0)
    self.assertEqual(max(support), 10.0)
    self.assertLen(support, 51)
    self.assertEqual(agent.gamma, 0.99)
    self.assertEqual(agent.update_horizon, 3)
    self.assertEqual(agent.min_replay_history, 20000)
    self.assertEqual(agent.update_period, 4)
    self.assertEqual(agent.target_update_period, 8000)
    self.assertEqual(agent.epsilon_train, 0.01)
    self.assertEqual(agent.epsilon_eval, 0.001)
    self.assertEqual(agent.epsilon_decay_period, 250000)
    self.assertEqual(agent._replay._max_capacity, 1000000)
    self.assertEqual(agent._replay._batch_size, 32)

  def testDefaultGinImplicitQuantile(self):
    """Test default ImplicitQuantile configuration using ICML gin."""
    logging.info('###### Training the Implicit Quantile agent #####')
    self._base_dir = os.path.join(
        '/tmp/dopamine_tests',
        datetime.datetime.utcnow().strftime('run_%Y_%m_%d_%H_%M_%S'),
    )
    logging.info('###### IQN base dir: %s', self._base_dir)
    gin_files = [
        'dopamine/jax/agents/'
        + 'implicit_quantile/configs/implicit_quantile.gin'
    ]
    gin_bindings = [
        'Runner.num_iterations=0',
    ]
    run_experiment.load_gin_configs(gin_files, gin_bindings)
    runner = run_experiment.Runner(
        self._base_dir,
        run_experiment.create_agent,
        atari_lib.create_atari_environment,
    )
    self.assertEqual(1000000, runner._agent._replay._max_capacity)
    shutil.rmtree(self._base_dir)

  def testOverrideGinImplicitQuantile(self):
    """Test ImplicitQuantile configuration overriding using IQN gin."""
    logging.info('###### Training the Implicit Quantile agent #####')
    self._base_dir = os.path.join(
        '/tmp/dopamine_tests',
        datetime.datetime.utcnow().strftime('run_%Y_%m_%d_%H_%M_%S'),
    )
    logging.info('###### IQN base dir: %s', self._base_dir)
    gin_files = [
        'dopamine/jax/agents/implicit_quantile/configs/'
        + 'implicit_quantile.gin'
    ]
    gin_bindings = [
        'Runner.num_iterations=0',
        'ReplayBuffer.max_capacity = 1000',
    ]
    run_experiment.load_gin_configs(gin_files, gin_bindings)
    runner = run_experiment.Runner(
        self._base_dir,
        run_experiment.create_agent,
        atari_lib.create_atari_environment,
    )
    self.assertEqual(1000, runner._agent._replay._max_capacity)
    shutil.rmtree(self._base_dir)


if __name__ == '__main__':
  tf.compat.v1.disable_v2_behavior()
  tf.test.main()

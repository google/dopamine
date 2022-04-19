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
"""Tests for dopamine.agents.dqn_agent."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil



from absl import flags
from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import atari_lib
from dopamine.utils import test_utils
import gin.tf
import mock
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS


class DQNAgentTest(tf.test.TestCase):

  def setUp(self):
    super(DQNAgentTest, self).setUp()
    self._test_subdir = os.path.join('/tmp/dopamine_tests', 'ckpts')
    shutil.rmtree(self._test_subdir, ignore_errors=True)
    os.makedirs(self._test_subdir)
    self.num_actions = 4
    self.min_replay_history = 6
    self.update_period = 2
    self.target_update_period = 4
    self.epsilon_decay_period = 90
    self.epsilon_train = 0.05
    self.observation_shape = dqn_agent.NATURE_DQN_OBSERVATION_SHAPE
    self.observation_dtype = dqn_agent.NATURE_DQN_DTYPE
    self.stack_size = dqn_agent.NATURE_DQN_STACK_SIZE
    self.zero_state = np.zeros(
        (1,) + self.observation_shape + (self.stack_size,))
    gin.bind_parameter('WrappedReplayBuffer.replay_capacity', 100)

  def _create_test_agent(self, sess, allow_partial_reload=False):
    stack_size = self.stack_size

    # This dummy network allows us to deterministically anticipate that
    # action 0 will be selected by an argmax.
    class MockDQNNetwork(tf.keras.Model):
      """The Keras network used in tests."""

      def __init__(self, num_actions, **kwargs):
        # This weights_initializer gives action 0 a higher weight, ensuring
        # that it gets picked by the argmax.
        super(MockDQNNetwork, self).__init__(**kwargs)
        weights_initializer = np.tile(
            np.arange(num_actions, 0, -1), (stack_size, 1))
        self.layer = tf.keras.layers.Dense(
            num_actions,
            kernel_initializer=tf.constant_initializer(weights_initializer),
            bias_initializer=tf.ones_initializer())

      def call(self, state):
        inputs = tf.constant(
            np.zeros((state.shape[0], stack_size)), dtype=tf.float32)
        return atari_lib.DQNNetworkType(self.layer((inputs)))

    agent = dqn_agent.DQNAgent(
        sess=sess,
        network=MockDQNNetwork,
        observation_shape=self.observation_shape,
        observation_dtype=self.observation_dtype,
        stack_size=self.stack_size,
        num_actions=self.num_actions,
        min_replay_history=self.min_replay_history,
        epsilon_fn=lambda w, x, y, z: 0.0,  # No exploration.
        update_period=self.update_period,
        target_update_period=self.target_update_period,
        epsilon_eval=0.0,  # No exploration during evaluation.
        allow_partial_reload=allow_partial_reload)
    # This ensures non-random action choices (since epsilon_eval = 0.0) and
    # skips the train_step.
    agent.eval_mode = True
    sess.run(tf.compat.v1.global_variables_initializer())
    return agent

  def testCreateAgentWithDefaults(self):
    # Verifies that we can create and train an agent with the default values.
    with tf.compat.v1.Session() as sess:
      agent = dqn_agent.DQNAgent(sess, num_actions=4)
      sess.run(tf.compat.v1.global_variables_initializer())
      observation = np.ones([84, 84, 1])
      agent.begin_episode(observation)
      agent.step(reward=1, observation=observation)
      agent.end_episode(reward=1)

  def testBeginEpisode(self):
    """Test the functionality of agent.begin_episode.

    Specifically, the action returned and its effect on state.
    """
    with tf.compat.v1.Session() as sess:
      agent = self._create_test_agent(sess)
      # We fill up the state with 9s. On calling agent.begin_episode the state
      # should be reset to all 0s.
      agent.state.fill(9)
      first_observation = np.ones(self.observation_shape + (1,))
      self.assertEqual(agent.begin_episode(first_observation), 0)
      # When the all-1s observation is received, it will be placed at the end of
      # the state.
      expected_state = self.zero_state
      expected_state[:, :, :, -1] = np.ones((1,) + self.observation_shape)
      self.assertAllEqual(agent.state, expected_state)
      self.assertAllEqual(agent._observation, first_observation[:, :, 0])
      # No training happens in eval mode.
      self.assertEqual(agent.training_steps, 0)

      # This will now cause training to happen.
      agent.eval_mode = False
      # Having a low replay memory add_count will prevent any of the
      # train/prefetch/sync ops from being called.
      agent._replay.memory.add_count = 0
      second_observation = np.ones(self.observation_shape + (1,)) * 2
      agent.begin_episode(second_observation)
      # The agent's state will be reset, so we will only be left with the all-2s
      # observation.
      expected_state[:, :, :, -1] = np.full((1,) + self.observation_shape, 2)
      self.assertAllEqual(agent.state, expected_state)
      self.assertAllEqual(agent._observation, second_observation[:, :, 0])
      # training_steps is incremented since we set eval_mode to False.
      self.assertEqual(agent.training_steps, 1)

  def testStepEval(self):
    """Test the functionality of agent.step() in eval mode.

    Specifically, the action returned, and confirm no training is happening.
    """
    with tf.compat.v1.Session() as sess:
      agent = self._create_test_agent(sess)
      base_observation = np.ones(self.observation_shape + (1,))
      # This will reset state and choose a first action.
      agent.begin_episode(base_observation)
      # We mock the replay buffer to verify how the agent interacts with it.
      agent._replay = test_utils.MockReplayBuffer()
      self.evaluate(tf.compat.v1.global_variables_initializer())

      expected_state = self.zero_state
      num_steps = 10
      for step in range(1, num_steps + 1):
        # We make observation a multiple of step for testing purposes (to
        # uniquely identify each observation).
        observation = base_observation * step
        self.assertEqual(agent.step(reward=1, observation=observation), 0)
        stack_pos = step - num_steps - 1
        if stack_pos >= -self.stack_size:
          expected_state[:, :, :, stack_pos] = np.full(
              (1,) + self.observation_shape, step)
      self.assertAllEqual(agent.state, expected_state)
      self.assertAllEqual(
          agent._last_observation,
          np.ones(self.observation_shape) * (num_steps - 1))
      self.assertAllEqual(agent._observation, observation[:, :, 0])
      # No training happens in eval mode.
      self.assertEqual(agent.training_steps, 0)
      # No transitions are added in eval mode.
      self.assertEqual(agent._replay.add.call_count, 0)

  def testStepTrain(self):
    """Test the functionality of agent.step() in train mode.

    Specifically, the action returned, and confirm training is happening.
    """
    with tf.compat.v1.Session() as sess:
      agent = self._create_test_agent(sess)
      agent.eval_mode = False
      base_observation = np.ones(self.observation_shape + (1,))
      # We mock the replay buffer to verify how the agent interacts with it.
      agent._replay = test_utils.MockReplayBuffer()
      self.evaluate(tf.compat.v1.global_variables_initializer())
      # This will reset state and choose a first action.
      agent.begin_episode(base_observation)
      observation = base_observation

      expected_state = self.zero_state
      num_steps = 10
      for step in range(1, num_steps + 1):
        # We make observation a multiple of step for testing purposes (to
        # uniquely identify each observation).
        last_observation = observation
        observation = base_observation * step
        self.assertEqual(agent.step(reward=1, observation=observation), 0)
        stack_pos = step - num_steps - 1
        if stack_pos >= -self.stack_size:
          expected_state[:, :, :, stack_pos] = np.full(
              (1,) + self.observation_shape, step)
        self.assertEqual(agent._replay.add.call_count, step)
        mock_args, _ = agent._replay.add.call_args
        self.assertAllEqual(last_observation[:, :, 0], mock_args[0])
        self.assertAllEqual(0, mock_args[1])  # Action selected.
        self.assertAllEqual(1, mock_args[2])  # Reward received.
        self.assertFalse(mock_args[3])  # is_terminal
      self.assertAllEqual(agent.state, expected_state)
      self.assertAllEqual(
          agent._last_observation,
          np.full(self.observation_shape, num_steps - 1))
      self.assertAllEqual(agent._observation, observation[:, :, 0])
      # We expect one more than num_steps because of the call to begin_episode.
      self.assertEqual(agent.training_steps, num_steps + 1)
      self.assertEqual(agent._replay.add.call_count, num_steps)

      agent.end_episode(reward=1)
      self.assertEqual(agent._replay.add.call_count, num_steps + 1)
      mock_args, _ = agent._replay.add.call_args
      self.assertAllEqual(observation[:, :, 0], mock_args[0])
      self.assertAllEqual(0, mock_args[1])  # Action selected.
      self.assertAllEqual(1, mock_args[2])  # Reward received.
      self.assertTrue(mock_args[3])  # is_terminal

  def testNonTupleObservationShape(self):
    with self.assertRaises(AssertionError):
      self.observation_shape = 84
      with tf.compat.v1.Session() as sess:
        _ = self._create_test_agent(sess)

  def _test_custom_shapes(self, shape, dtype, stack_size):
    self.observation_shape = shape
    self.observation_dtype = dtype
    self.stack_size = stack_size
    self.zero_state = np.zeros((1,) + shape + (stack_size,))
    with tf.compat.v1.Session() as sess:
      agent = self._create_test_agent(sess)
      agent.eval_mode = False
      base_observation = np.ones(self.observation_shape + (1,))
      # We mock the replay buffer to verify how the agent interacts with it.
      agent._replay = test_utils.MockReplayBuffer()
      self.evaluate(tf.compat.v1.global_variables_initializer())
      # This will reset state and choose a first action.
      agent.begin_episode(base_observation)
      observation = base_observation

      expected_state = self.zero_state
      num_steps = 10
      for step in range(1, num_steps + 1):
        # We make observation a multiple of step for testing purposes (to
        # uniquely identify each observation).
        last_observation = observation
        observation = base_observation * step
        self.assertEqual(agent.step(reward=1, observation=observation), 0)
        stack_pos = step - num_steps - 1
        if stack_pos >= -self.stack_size:
          expected_state[..., stack_pos] = np.full(
              (1,) + self.observation_shape, step)
        self.assertEqual(agent._replay.add.call_count, step)
        mock_args, _ = agent._replay.add.call_args
        self.assertAllEqual(last_observation[..., 0], mock_args[0])
        self.assertAllEqual(0, mock_args[1])  # Action selected.
        self.assertAllEqual(1, mock_args[2])  # Reward received.
        self.assertFalse(mock_args[3])  # is_terminal
      self.assertAllEqual(agent.state, expected_state)
      self.assertAllEqual(
          agent._last_observation,
          np.full(self.observation_shape, num_steps - 1))
      self.assertAllEqual(agent._observation, observation[..., 0])
      # We expect one more than num_steps because of the call to begin_episode.
      self.assertEqual(agent.training_steps, num_steps + 1)
      self.assertEqual(agent._replay.add.call_count, num_steps)

      agent.end_episode(reward=1)
      self.assertEqual(agent._replay.add.call_count, num_steps + 1)
      mock_args, _ = agent._replay.add.call_args
      self.assertAllEqual(observation[..., 0], mock_args[0])
      self.assertAllEqual(0, mock_args[1])  # Action selected.
      self.assertAllEqual(1, mock_args[2])  # Reward received.
      self.assertTrue(mock_args[3])  # is_terminal

  def testStepTrainCustomObservationShapes(self):
    custom_shapes = [(1,), (4, 4), (6, 1), (1, 6), (1, 1, 6), (6, 6, 6, 6)]
    for shape in custom_shapes:
      self._test_custom_shapes(shape, tf.uint8, 1)

  def testStepTrainCustomTypes(self):
    custom_types = [tf.float32, tf.uint8, tf.int64]
    for dtype in custom_types:
      self._test_custom_shapes((4, 4), dtype, 1)

  def testStepTrainCustomStackSizes(self):
    custom_stack_sizes = [1, 4, 8]
    for stack_size in custom_stack_sizes:
      self._test_custom_shapes((4, 4), tf.uint8, stack_size)

  def testLinearlyDecayingEpsilon(self):
    """Test the functionality of the linearly_decaying_epsilon function."""
    decay_period = 100
    warmup_steps = 6
    epsilon = 0.1
    steps_schedule = [
        (0, 1.0),  # step < warmup_steps
        (16, 0.91),  # bonus = 0.9 * 90 / 100 = 0.81
        (decay_period + warmup_steps + 1, epsilon)]  # step > decay+warmup
    for step, expected_epsilon in steps_schedule:
      self.assertNear(dqn_agent.linearly_decaying_epsilon(decay_period,
                                                          step,
                                                          warmup_steps,
                                                          epsilon),
                      expected_epsilon, 0.01)

  def testBundlingWithNonexistentDirectory(self):
    with tf.compat.v1.Session() as sess:
      agent = self._create_test_agent(sess)
      self.assertIsNone(agent.bundle_and_checkpoint('/does/not/exist', 1))

  def testUnbundlingWithFailingReplayBuffer(self):
    with tf.compat.v1.Session() as sess:
      agent = self._create_test_agent(sess)
      bundle = {}
      # The ReplayBuffer will throw an exception since it is not able to load
      # the expected files, which will cause the unbundle() method to return
      # False.
      self.assertFalse(agent.unbundle(self._test_subdir, 1729, bundle))

  def testUnbundlingWithNoBundleDictionary(self):
    with tf.compat.v1.Session() as sess:
      agent = self._create_test_agent(sess)
      agent._replay = mock.Mock()
      self.assertFalse(agent.unbundle(self._test_subdir, 1729, None))

  def testPartialUnbundling(self):
    with tf.compat.v1.Session() as sess:
      agent = self._create_test_agent(sess, allow_partial_reload=True)
      # These values don't reflect the actual types of these attributes, but are
      # used merely for facility of testing.
      agent.state = 'state'
      agent.training_steps = 'training_steps'
      iteration_number = 1729
      _ = agent.bundle_and_checkpoint(self._test_subdir, iteration_number)
      # Both the ReplayBuffer and bundle_dictionary checks will fail,
      # but this will be ignored since we're allowing partial reloads.
      self.assertTrue(agent.unbundle(self._test_subdir, 1729, None))

  def testBundling(self):
    with tf.compat.v1.Session() as sess:
      agent = self._create_test_agent(sess)
      # These values don't reflect the actual types of these attributes, but are
      # used merely for facility of testing.
      agent.state = 'state'
      agent._replay = mock.Mock()
      agent.training_steps = 'training_steps'
      iteration_number = 1729
      bundle = agent.bundle_and_checkpoint(self._test_subdir, iteration_number)
      keys = ['state', 'training_steps']
      for key in keys:
        self.assertIn(key, bundle)
        self.assertEqual(key, bundle[key])

      agent.unbundle(self._test_subdir, iteration_number, bundle)
      for key in keys:
        self.assertEqual(key, agent.__dict__[key])

  def testSyncOpWithNameScopes(self):
    num_agents = 5
    with tf.compat.v1.Session() as sess:
      agents = []
      for i in range(num_agents):
        with tf.name_scope('agent_{}'.format(i)):
          agents.append(self._create_test_agent(sess))

      # Checks that the agents have two ops each.
      ops = []
      for agent in agents:
        self.assertLen(agent._sync_qt_ops, 2)
        ops.extend(agent._sync_qt_ops)

      # Assigns the bias of the first agent to 0.
      ops.append(agents[0].online_convnet.layer.bias.assign(tf.zeros(4)))
      # Runs twice to make sure zeros are propagated to online and target.
      sess.run(ops)
      sess.run(ops)

      biases_0 = sess.run([
          agents[0].target_convnet.layer.bias,
          agents[0].online_convnet.layer.bias])
      self.assertAllEqual(biases_0[0], tf.zeros(4))
      self.assertAllEqual(biases_0[1], biases_0[0])

      # Checks that other agents have not been affected.
      biases = sess.run([agt.target_convnet.layer.bias for agt in agents[1:]])
      for bias in biases:
        self.assertNotAllEqual(bias, biases_0[0])


if __name__ == '__main__':
  tf.compat.v1.disable_v2_behavior()
  tf.test.main()

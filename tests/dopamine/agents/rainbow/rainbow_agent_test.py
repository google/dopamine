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
"""Tests for dopamine.agents.rainbow.rainbow_agent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from dopamine.agents.dqn import dqn_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import atari_lib
from dopamine.utils import test_utils
import numpy as np
import tensorflow as tf


class ProjectDistributionTest(tf.test.TestCase):

  def testInconsistentSupportsAndWeightsParameters(self):
    supports = tf.constant([[0, 2, 4, 6, 8], [3, 4, 5, 6, 7]], dtype=tf.float32)
    weights = tf.constant(
        [[0.1, 0.2, 0.3, 0.2], [0.1, 0.2, 0.3, 0.2]], dtype=tf.float32)
    target_support = tf.constant([4, 5, 6, 7, 8], dtype=tf.float32)
    with self.assertRaisesRegexp(ValueError, 'are incompatible'):
      rainbow_agent.project_distribution(supports, weights, target_support)

  def testInconsistentSupportsAndWeightsWithPlaceholders(self):
    supports = [[0, 2, 4, 6, 8], [3, 4, 5, 6, 7]]
    supports_ph = tf.compat.v1.placeholder(tf.float32, None)
    weights = [[0.1, 0.2, 0.3, 0.2], [0.1, 0.2, 0.3, 0.2]]
    weights_ph = tf.compat.v1.placeholder(tf.float32, None)
    target_support = [4, 5, 6, 7, 8]
    target_support_ph = tf.compat.v1.placeholder(tf.float32, None)
    projection = rainbow_agent.project_distribution(
        supports_ph, weights_ph, target_support_ph, validate_args=True)
    with self.test_session() as sess:
      tf.compat.v1.global_variables_initializer().run()
      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                   'assertion failed'):
        sess.run(
            projection,
            feed_dict={
                supports_ph: supports,
                weights_ph: weights,
                target_support_ph: target_support
            })

  def testInconsistentSupportsAndTargetSupportParameters(self):
    supports = tf.constant([[0, 2, 4, 6, 8], [3, 4, 5, 6, 7]], dtype=tf.float32)
    weights = tf.constant(
        [[0.1, 0.2, 0.3, 0.2, 0.2], [0.1, 0.2, 0.3, 0.2, 0.2]],
        dtype=tf.float32)
    target_support = tf.constant([4, 5, 6], dtype=tf.float32)
    with self.assertRaisesRegexp(ValueError, 'are incompatible'):
      rainbow_agent.project_distribution(supports, weights, target_support)

  def testInconsistentSupportsAndTargetSupportWithPlaceholders(self):
    supports = [[0, 2, 4, 6, 8], [3, 4, 5, 6, 7]]
    supports_ph = tf.compat.v1.placeholder(tf.float32, None)
    weights = [[0.1, 0.2, 0.3, 0.2, 0.2], [0.1, 0.2, 0.3, 0.2, 0.2]]
    weights_ph = tf.compat.v1.placeholder(tf.float32, None)
    target_support = [4, 5, 6]
    target_support_ph = tf.compat.v1.placeholder(tf.float32, None)
    projection = rainbow_agent.project_distribution(
        supports_ph, weights_ph, target_support_ph, validate_args=True)
    with self.test_session() as sess:
      tf.compat.v1.global_variables_initializer().run()
      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                   'assertion failed'):
        sess.run(
            projection,
            feed_dict={
                supports_ph: supports,
                weights_ph: weights,
                target_support_ph: target_support
            })

  def testZeroDimensionalTargetSupport(self):
    supports = tf.constant([[0, 2, 4, 6, 8], [3, 4, 5, 6, 7]], dtype=tf.float32)
    weights = tf.constant(
        [[0.1, 0.2, 0.3, 0.2, 0.2], [0.1, 0.2, 0.3, 0.2, 0.2]],
        dtype=tf.float32)
    target_support = tf.constant(3, dtype=tf.float32)
    with self.assertRaisesRegexp(ValueError, 'Index out of range'):
      rainbow_agent.project_distribution(supports, weights, target_support)

  def testZeroDimensionalTargetSupportWithPlaceholders(self):
    supports = [[0, 2, 4, 6, 8], [3, 4, 5, 6, 7]]
    supports_ph = tf.compat.v1.placeholder(tf.float32, None)
    weights = [[0.1, 0.2, 0.3, 0.2, 0.2], [0.1, 0.2, 0.3, 0.2, 0.2]]
    weights_ph = tf.compat.v1.placeholder(tf.float32, None)
    target_support = 3
    target_support_ph = tf.compat.v1.placeholder(tf.float32, None)
    projection = rainbow_agent.project_distribution(
        supports_ph, weights_ph, target_support_ph, validate_args=True)
    with self.test_session() as sess:
      tf.compat.v1.global_variables_initializer().run()
      with self.assertRaises(tf.errors.InvalidArgumentError):
        sess.run(
            projection,
            feed_dict={
                supports_ph: supports,
                weights_ph: weights,
                target_support_ph: target_support
            })

  def testMultiDimensionalTargetSupport(self):
    supports = tf.constant([[0, 2, 4, 6, 8], [3, 4, 5, 6, 7]], dtype=tf.float32)
    weights = tf.constant(
        [[0.1, 0.2, 0.3, 0.2, 0.2], [0.1, 0.2, 0.3, 0.2, 0.2]],
        dtype=tf.float32)
    target_support = tf.constant([[3]], dtype=tf.float32)
    with self.assertRaisesRegexp(ValueError, 'out of bounds'):
      rainbow_agent.project_distribution(supports, weights, target_support)

  def testMultiDimensionalTargetSupportWithPlaceholders(self):
    supports = [[0, 2, 4, 6, 8], [3, 4, 5, 6, 7]]
    supports_ph = tf.compat.v1.placeholder(tf.float32, None)
    weights = [[0.1, 0.2, 0.3, 0.2, 0.2], [0.1, 0.2, 0.3, 0.2, 0.2]]
    weights_ph = tf.compat.v1.placeholder(tf.float32, None)
    target_support = [[3]]
    target_support_ph = tf.compat.v1.placeholder(tf.float32, None)
    projection = rainbow_agent.project_distribution(
        supports_ph, weights_ph, target_support_ph, validate_args=True)
    with self.test_session() as sess:
      tf.compat.v1.global_variables_initializer().run()
      with self.assertRaises(tf.errors.InvalidArgumentError):
        sess.run(
            projection,
            feed_dict={
                supports_ph: supports,
                weights_ph: weights,
                target_support_ph: target_support
            })

  def testProjectWithNonMonotonicTargetSupport(self):
    supports = tf.constant([[0, 2, 4, 6, 8], [3, 4, 5, 6, 7]], dtype=tf.float32)
    weights = tf.constant(
        [[0.1, 0.2, 0.3, 0.2, 0.2], [0.1, 0.2, 0.3, 0.2, 0.2]],
        dtype=tf.float32)
    target_support = tf.constant([8, 7, 6, 5, 4], dtype=tf.float32)
    projection = rainbow_agent.project_distribution(
        supports, weights, target_support, validate_args=True)
    with self.test_session() as sess:
      tf.compat.v1.global_variables_initializer().run()
      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                   'assertion failed'):
        sess.run(projection)

  def testProjectNewSupportHasInconsistentDeltask(self):
    supports = tf.constant([[0, 2, 4, 6, 8], [3, 4, 5, 6, 7]], dtype=tf.float32)
    weights = tf.constant(
        [[0.1, 0.2, 0.3, 0.2, 0.2], [0.1, 0.2, 0.3, 0.2, 0.2]],
        dtype=tf.float32)
    target_support = tf.constant([3, 4, 6, 7, 8], dtype=tf.float32)
    projection = rainbow_agent.project_distribution(
        supports, weights, target_support, validate_args=True)
    with self.test_session() as sess:
      tf.compat.v1.global_variables_initializer().run()
      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                   'assertion failed'):
        sess.run(projection)

  def testProjectSingleIdenticalDistribution(self):
    supports = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.float32)
    expected_weights = [0.1, 0.2, 0.1, 0.3, 0.3]
    weights = tf.constant([expected_weights], dtype=tf.float32)
    target_support = tf.constant([0, 1, 2, 3, 4], dtype=tf.float32)
    projection = rainbow_agent.project_distribution(supports, weights,
                                                    target_support)
    with self.test_session() as sess:
      tf.compat.v1.global_variables_initializer().run()
      projection_ = sess.run(projection)
      self.assertAllClose([expected_weights], projection_)

  def testProjectSingleDifferentDistribution(self):
    supports = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.float32)
    weights = tf.constant([[0.1, 0.2, 0.1, 0.3, 0.3]], dtype=tf.float32)
    target_support = tf.constant([3, 4, 5, 6, 7], dtype=tf.float32)
    projection = rainbow_agent.project_distribution(supports, weights,
                                                    target_support)
    expected_projection = [[0.7, 0.3, 0.0, 0.0, 0.0]]
    with self.test_session() as sess:
      tf.compat.v1.global_variables_initializer().run()
      projection_ = sess.run(projection)
      self.assertAllClose(expected_projection, projection_)

  def testProjectFromNonMonotonicSupport(self):
    supports = tf.constant([[4, 3, 2, 1, 0]], dtype=tf.float32)
    weights = tf.constant([[0.1, 0.2, 0.1, 0.3, 0.3]], dtype=tf.float32)
    target_support = tf.constant([3, 4, 5, 6, 7], dtype=tf.float32)
    projection = rainbow_agent.project_distribution(supports, weights,
                                                    target_support)
    expected_projection = [[0.9, 0.1, 0.0, 0.0, 0.0]]
    with self.test_session() as sess:
      tf.compat.v1.global_variables_initializer().run()
      projection_ = sess.run(projection)
      self.assertAllClose(expected_projection, projection_)

  def testExampleFromCodeComments(self):
    supports = tf.constant([[0, 2, 4, 6, 8], [1, 3, 4, 5, 6]], dtype=tf.float32)
    weights = tf.constant(
        [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.2, 0.5, 0.1, 0.1]],
        dtype=tf.float32)
    target_support = tf.constant([4, 5, 6, 7, 8], dtype=tf.float32)
    projection = rainbow_agent.project_distribution(supports, weights,
                                                    target_support)
    expected_projections = [[0.8, 0.0, 0.1, 0.0, 0.1],
                            [0.8, 0.1, 0.1, 0.0, 0.0]]
    with self.test_session() as sess:
      tf.compat.v1.global_variables_initializer().run()
      projection_ = sess.run(projection)
      self.assertAllClose(expected_projections, projection_)

  def testProjectBatchOfDifferentDistributions(self):
    supports = tf.constant(
        [[0, 2, 4, 6, 8], [0, 1, 2, 3, 4], [3, 4, 5, 6, 7]], dtype=tf.float32)
    weights = tf.constant(
        [[0.1, 0.2, 0.3, 0.2, 0.2], [0.1, 0.2, 0.1, 0.3, 0.3],
         [0.1, 0.2, 0.3, 0.2, 0.2]],
        dtype=tf.float32)
    target_support = tf.constant([3, 4, 5, 6, 7], dtype=tf.float32)
    projection = rainbow_agent.project_distribution(supports, weights,
                                                    target_support)
    expected_projections = [[0.3, 0.3, 0.0, 0.2,
                             0.2], [0.7, 0.3, 0.0, 0.0, 0.0],
                            [0.1, 0.2, 0.3, 0.2, 0.2]]
    with self.test_session() as sess:
      tf.compat.v1.global_variables_initializer().run()
      projection_ = sess.run(projection)
      self.assertAllClose(expected_projections, projection_)

  def testUsingPlaceholders(self):
    supports = [[0, 2, 4, 6, 8], [0, 1, 2, 3, 4], [3, 4, 5, 6, 7]]
    supports_ph = tf.compat.v1.placeholder(tf.float32, None)
    weights = [[0.1, 0.2, 0.3, 0.2, 0.2], [0.1, 0.2, 0.1, 0.3, 0.3],
               [0.1, 0.2, 0.3, 0.2, 0.2]]
    weights_ph = tf.compat.v1.placeholder(tf.float32, None)
    target_support = [3, 4, 5, 6, 7]
    target_support_ph = tf.compat.v1.placeholder(tf.float32, None)
    projection = rainbow_agent.project_distribution(supports_ph, weights_ph,
                                                    target_support_ph)
    expected_projections = [[0.3, 0.3, 0.0, 0.2,
                             0.2], [0.7, 0.3, 0.0, 0.0, 0.0],
                            [0.1, 0.2, 0.3, 0.2, 0.2]]
    with self.test_session() as sess:
      tf.compat.v1.global_variables_initializer().run()
      projection_ = sess.run(
          projection,
          feed_dict={
              supports_ph: supports,
              weights_ph: weights,
              target_support_ph: target_support
          })
      self.assertAllClose(expected_projections, projection_)

  def testProjectBatchOfDifferentDistributionsWithLargerDelta(self):
    supports = tf.constant(
        [[0, 2, 4, 6, 8], [8, 9, 10, 12, 14]], dtype=tf.float32)
    weights = tf.constant(
        [[0.1, 0.2, 0.2, 0.2, 0.3], [0.1, 0.2, 0.4, 0.1, 0.2]],
        dtype=tf.float32)
    target_support = tf.constant([0, 4, 8, 12, 16], dtype=tf.float32)
    projection = rainbow_agent.project_distribution(supports, weights,
                                                    target_support)
    expected_projections = [[0.2, 0.4, 0.4, 0.0, 0.0],
                            [0.0, 0.0, 0.45, 0.45, 0.1]]
    with self.test_session() as sess:
      tf.compat.v1.global_variables_initializer().run()
      projection_ = sess.run(projection)
      self.assertAllClose(expected_projections, projection_)


class RainbowAgentTest(tf.test.TestCase):

  def setUp(self):
    super(RainbowAgentTest, self).setUp()
    self._num_actions = 4
    self._num_atoms = 5
    self._vmax = 7.
    self._min_replay_history = 32
    self._epsilon_decay_period = 90
    self.observation_shape = dqn_agent.NATURE_DQN_OBSERVATION_SHAPE
    self.observation_dtype = dqn_agent.NATURE_DQN_DTYPE
    self.stack_size = dqn_agent.NATURE_DQN_STACK_SIZE
    self.zero_state = np.zeros(
        (1,) + self.observation_shape + (self.stack_size,))

  def _create_test_agent(self, sess):
    stack_size = self.stack_size
    # This dummy network allows us to deterministically anticipate that
    # action 0 will be selected by an argmax.

    # In Rainbow we are dealing with a distribution over Q-values,
    # which are represented as num_atoms bins, ranging from -vmax to vmax.
    # The output layer will have num_actions * num_atoms elements,
    # so each group of num_atoms weights represent the logits for a
    # particular action. By setting 1s everywhere, except for the first
    # num_atoms (representing the logits for the first action), which are
    # set to np.arange(num_atoms), we are ensuring that the first action
    # places higher weight on higher Q-values; this results in the first
    # action being chosen.
    class MockRainbowNetwork(tf.keras.Model):
      """Custom tf.keras.Model used in tests."""

      def __init__(self, num_actions, num_atoms, support, **kwargs):
        super(MockRainbowNetwork, self).__init__(**kwargs)
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.support = support
        first_row = np.tile(np.ones(self.num_atoms), self.num_actions - 1)
        first_row = np.concatenate((np.arange(self.num_atoms), first_row))
        bottom_rows = np.tile(
            np.ones(self.num_actions * self.num_atoms), (stack_size - 1, 1))
        weights_initializer = np.concatenate(([first_row], bottom_rows))
        self.layer = tf.keras.layers.Dense(
            self.num_actions * self.num_atoms,
            kernel_initializer=tf.constant_initializer(weights_initializer),
            bias_initializer=tf.ones_initializer())

      def call(self, state):
        inputs = tf.constant(
            np.zeros((state.shape[0], stack_size)), dtype=tf.float32)
        net = self.layer(inputs)
        logits = tf.reshape(net, [-1, self.num_actions, self.num_atoms])
        probabilities = tf.keras.activations.softmax(logits)
        qs = tf.reduce_sum(self.support * probabilities, axis=2)
        return atari_lib.RainbowNetworkType(qs, logits, probabilities)

    agent = rainbow_agent.RainbowAgent(
        sess=sess,
        network=MockRainbowNetwork,
        num_actions=self._num_actions,
        num_atoms=self._num_atoms,
        vmax=self._vmax,
        min_replay_history=self._min_replay_history,
        epsilon_fn=lambda w, x, y, z: 0.0,  # No exploration.
        epsilon_eval=0.0,
        epsilon_decay_period=self._epsilon_decay_period)
    # This ensures non-random action choices (since epsilon_eval = 0.0) and
    # skips the train_step.
    agent.eval_mode = True
    sess.run(tf.compat.v1.global_variables_initializer())
    return agent

  def testCreateAgentWithDefaults(self):
    # Verifies that we can create and train an agent with the default values.
    with tf.compat.v1.Session() as sess:
      agent = rainbow_agent.RainbowAgent(sess, num_actions=4)
      sess.run(tf.compat.v1.global_variables_initializer())
      observation = np.ones([84, 84, 1])
      agent.begin_episode(observation)
      agent.step(reward=1, observation=observation)
      agent.end_episode(reward=1)

  def testShapesAndValues(self):
    with tf.compat.v1.Session() as sess:
      agent = self._create_test_agent(sess)
      self.assertEqual(agent._support.shape[0], self._num_atoms)
      self.assertEqual(
          self.evaluate(tf.reduce_min(agent._support)), -self._vmax)
      self.assertEqual(self.evaluate(tf.reduce_max(agent._support)), self._vmax)
      self.assertEqual(agent._net_outputs.logits.shape,
                       (1, self._num_actions, self._num_atoms))
      self.assertEqual(agent._net_outputs.probabilities.shape,
                       agent._net_outputs.logits.shape)
      self.assertEqual(agent._replay_net_outputs.logits.shape[1],
                       self._num_actions)
      self.assertEqual(agent._replay_net_outputs.logits.shape[2],
                       self._num_atoms)
      self.assertEqual(agent._replay_next_target_net_outputs.logits.shape[1],
                       self._num_actions)
      self.assertEqual(agent._replay_next_target_net_outputs.logits.shape[2],
                       self._num_atoms)
      self.assertEqual(agent._net_outputs.q_values.shape,
                       (1, self._num_actions))

  def testBeginEpisode(self):
    """Tests the functionality of agent.begin_episode.

    Specifically, the action returned and its effect on the state.
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
    """Tests the functionality of agent.step() in eval mode.

    Specifically, the action returned, and confirms that no training happens.
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

    Specifically, the action returned, and confirms training is happening.
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
          np.full(self.observation_shape, num_steps - 1))
      self.assertAllEqual(agent._observation, observation[:, :, 0])
      # We expect one more than num_steps because of the call to begin_episode.
      self.assertEqual(agent.training_steps, num_steps + 1)
      self.assertEqual(agent._replay.add.call_count, num_steps)

      agent.end_episode(reward=1)
      self.assertEqual(agent._replay.add.call_count, num_steps + 1)

  def testStoreTransitionWithUniformSampling(self):
    with tf.compat.v1.Session() as sess:
      agent = rainbow_agent.RainbowAgent(
          sess, num_actions=4, replay_scheme='uniform')
      dummy_frame = np.zeros((84, 84))
      # Adding transitions with default, 10., default priorities.
      agent._store_transition(dummy_frame, 0, 0, False)
      agent._store_transition(dummy_frame, 0, 0, False, 10.)
      agent._store_transition(dummy_frame, 0, 0, False)
      returned_priorities = agent._replay.memory.get_priority(
          np.arange(self.stack_size - 1, self.stack_size + 2, dtype=np.int32))
      expected_priorities = [1., 10., 1.]
      self.assertAllEqual(returned_priorities, expected_priorities)

  def testStoreTransitionWithPrioritizedSamplingy(self):
    with tf.compat.v1.Session() as sess:
      agent = rainbow_agent.RainbowAgent(
          sess, num_actions=4, replay_scheme='prioritized')
      dummy_frame = np.zeros((84, 84))
      # Adding transitions with default, 10., default priorities.
      agent._store_transition(dummy_frame, 0, 0, False)
      agent._store_transition(dummy_frame, 0, 0, False, 10.)
      agent._store_transition(dummy_frame, 0, 0, False)
      returned_priorities = agent._replay.memory.get_priority(
          np.arange(self.stack_size - 1, self.stack_size + 2, dtype=np.int32))
      expected_priorities = [1., 10., 10.]
      self.assertAllEqual(returned_priorities, expected_priorities)


if __name__ == '__main__':
  tf.compat.v1.disable_v2_behavior()
  tf.test.main()

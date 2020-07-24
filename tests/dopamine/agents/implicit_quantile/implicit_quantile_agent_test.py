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
"""Tests for implicit quantile agent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from dopamine.agents.dqn import dqn_agent
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.discrete_domains import atari_lib
import numpy as np
import tensorflow as tf


class ImplicitQuantileAgentTest(tf.test.TestCase):

  def setUp(self):
    super(ImplicitQuantileAgentTest, self).setUp()
    self._num_actions = 4
    self.observation_shape = dqn_agent.NATURE_DQN_OBSERVATION_SHAPE
    self.observation_dtype = dqn_agent.NATURE_DQN_DTYPE
    self.stack_size = dqn_agent.NATURE_DQN_STACK_SIZE
    self.ones_state = np.ones(
        (1,) + self.observation_shape + (self.stack_size,))

  def _create_test_agent(self, sess):
    # This dummy network allows us to deterministically anticipate that the
    # state-action-quantile outputs will be equal to sum of the
    # corresponding quantile inputs.
    # State/Quantile shapes will be k x 1, (N x batch_size) x 1,
    # or (N' x batch_size) x 1.

    class MockImplicitQuantileNetwork(tf.keras.Model):
      """Custom tf.keras.Model used in tests."""

      def __init__(self, num_actions, quantile_embedding_dim, **kwargs):
        # This weights_initializer gives action 0 a higher weight, ensuring
        # that it gets picked by the argmax.
        super(MockImplicitQuantileNetwork, self).__init__(**kwargs)
        self.num_actions = num_actions
        self.layer = tf.keras.layers.Dense(
            self.num_actions, kernel_initializer=tf.ones_initializer(),
            bias_initializer=tf.zeros_initializer())

      def call(self, state, num_quantiles):
        batch_size = state.get_shape().as_list()[0]
        inputs = tf.constant(
            np.ones((batch_size*num_quantiles, self.num_actions)),
            dtype=tf.float32)
        quantiles_shape = [num_quantiles * batch_size, 1]
        quantiles = tf.ones(quantiles_shape)
        return atari_lib.ImplicitQuantileNetworkType(self.layer(inputs),
                                                     quantiles)

    agent = implicit_quantile_agent.ImplicitQuantileAgent(
        sess=sess,
        network=MockImplicitQuantileNetwork,
        num_actions=self._num_actions,
        kappa=1.0,
        num_tau_samples=2,
        num_tau_prime_samples=3,
        num_quantile_samples=4)
    # This ensures non-random action choices (since epsilon_eval = 0.0) and
    # skips the train_step.
    agent.eval_mode = True
    sess.run(tf.compat.v1.global_variables_initializer())
    return agent

  def testCreateAgentWithDefaults(self):
    # Verifies that we can create and train an agent with the default values.
    with self.test_session(use_gpu=False) as sess:
      agent = implicit_quantile_agent.ImplicitQuantileAgent(sess, num_actions=4)
      sess.run(tf.compat.v1.global_variables_initializer())
      observation = np.ones([84, 84, 1])
      agent.begin_episode(observation)
      agent.step(reward=1, observation=observation)
      agent.end_episode(reward=1)

  def testShapes(self):
    with self.test_session(use_gpu=False) as sess:
      agent = self._create_test_agent(sess)

      # Replay buffer batch size:
      self.assertEqual(agent._replay.batch_size, 32)

      # quantile values, q-values, q-argmax at sample action time:
      self.assertEqual(agent._net_outputs.quantile_values.shape[0],
                       agent.num_quantile_samples)
      self.assertEqual(agent._net_outputs.quantile_values.shape[1],
                       agent.num_actions)
      self.assertEqual(agent._q_values.shape[0], agent.num_actions)

      # Check the setting of num_actions.
      self.assertEqual(self._num_actions, agent.num_actions)

      # input quantiles, quantile values, and output q-values at loss
      # computation time.
      self.assertEqual(agent._replay_net_quantile_values.shape[0],
                       agent.num_tau_samples * agent._replay.batch_size)
      self.assertEqual(agent._replay_net_quantile_values.shape[1],
                       agent.num_actions)

      self.assertEqual(agent._replay_net_target_quantile_values.shape[0],
                       agent.num_tau_prime_samples * agent._replay.batch_size)
      self.assertEqual(agent._replay_net_target_quantile_values.shape[1],
                       agent.num_actions)

      self.assertEqual(agent._replay_net_target_q_values.shape[0],
                       agent._replay.batch_size)
      self.assertEqual(agent._replay_net_target_q_values.shape[1],
                       agent.num_actions)

  def test_q_value_computation(self):
    with self.test_session(use_gpu=False) as sess:
      agent = self._create_test_agent(sess)
      quantiles = np.ones(agent.num_quantile_samples)
      q_value = np.sum(quantiles)
      quantiles = quantiles.reshape([agent.num_quantile_samples, 1])
      state = self.ones_state
      feed_dict = {agent.state_ph: state}

      q_values, q_argmax = sess.run([agent._q_values, agent._q_argmax],
                                    feed_dict)

      q_values_arr = np.ones([agent.num_actions]) * q_value
      self.assertAllEqual(q_values, q_values_arr)
      self.assertEqual(q_argmax, 0)

      q_values_target = sess.run(agent._replay_net_target_q_values, feed_dict)

      batch_size = agent._replay.batch_size

      for i in range(batch_size):
        for j in range(agent.num_actions):
          self.assertEqual(q_values_target[i][j], q_values[j])

  def test_replay_quantile_value_computation(self):
    with self.test_session(use_gpu=False) as sess:
      agent = self._create_test_agent(sess)

      replay_quantile_vals, replay_target_quantile_vals = sess.run(
          [agent._replay_net_quantile_values,
           agent._replay_net_target_quantile_values])

      batch_size = agent._replay.batch_size
      replay_quantile_vals = replay_quantile_vals.reshape([
          agent.num_tau_samples, batch_size, agent.num_actions])
      replay_target_quantile_vals = replay_target_quantile_vals.reshape([
          agent.num_tau_prime_samples, batch_size, agent.num_actions])
      for i in range(agent.num_tau_samples):
        for j in range(agent._replay.batch_size):
          self.assertEqual(replay_quantile_vals[i][j][0], agent.num_actions)

      for i in range(agent.num_tau_prime_samples):
        for j in range(agent._replay.batch_size):
          self.assertEqual(replay_target_quantile_vals[i][j][0],
                           agent.num_actions)


if __name__ == '__main__':
  tf.compat.v1.disable_v2_behavior()
  tf.test.main()

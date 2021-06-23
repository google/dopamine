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
"""Tests for dopamine.jax.agents.dqn_agent."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

from absl import flags
from absl.testing import absltest
from dopamine.discrete_domains import atari_lib
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.utils import test_utils
from flax import linen as nn
import gin
import jax.numpy as jnp
import mock
import numpy as onp

FLAGS = flags.FLAGS


class DQNAgentTest(absltest.TestCase):

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
    self.observation_dtype = jnp.uint8
    self.stack_size = dqn_agent.NATURE_DQN_STACK_SIZE
    self.zero_state = onp.zeros(self.observation_shape + (self.stack_size,))
    gin.bind_parameter('OutOfGraphReplayBuffer.replay_capacity', 100)
    gin.bind_parameter('OutOfGraphReplayBuffer.batch_size', 2)

  def _create_test_agent(self, allow_partial_reload=False):

    # This dummy network allows us to deterministically anticipate that
    # action 0 will be selected by an argmax.
    class MockDQNNetwork(nn.Module):
      """The Jax network used in tests."""
      num_actions: int
      inputs_preprocessed: bool = False

      @nn.compact
      def __call__(self, x):
        # This weights_initializer gives action 0 a higher weight, ensuring
        # that it gets picked by the argmax.
        def custom_init(key, shape, dtype=jnp.float32):
          del key
          to_pick_first_action = onp.zeros(shape, dtype)
          to_pick_first_action[:, 0] = 1
          return to_pick_first_action
        x = x.astype(jnp.float32)
        x = x.reshape((-1))  # flatten
        x = nn.Dense(features=self.num_actions,
                     kernel_init=custom_init,
                     bias_init=nn.initializers.ones)(x)
        return atari_lib.DQNNetworkType(x)

    agent = dqn_agent.JaxDQNAgent(
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
    return agent

  def testCreateAgentWithDefaults(self):
    # Verifies that we can create and train an agent with the default values.
    agent = dqn_agent.JaxDQNAgent(num_actions=4)
    observation = onp.ones([84, 84, 1])
    agent.begin_episode(observation)
    agent.step(reward=1, observation=observation)
    agent.end_episode(reward=1)

  def testPreprocessFnParam(self):

    # Mock a regular network so we can inspect its constructor parameters.
    class MockNetwork(object):

      def __init__(self, num_actions=3, inputs_preprocessed=False):
        self.num_actions = num_actions
        self.inputs_preprocessed = inputs_preprocessed

    # We create a subclass of JaxDQNAgent so we can override
    # _build_networks_and_optimizer and avoid network creation.
    class MockDQNAgent(dqn_agent.JaxDQNAgent):

      def _build_networks_and_optimizer(self):
        pass

    # Test with the default setting for preprocess_fn (None).
    agent = MockDQNAgent(num_actions=4, network=MockNetwork)
    self.assertEqual(4, agent.network_def.num_actions)
    self.assertFalse(agent.network_def.inputs_preprocessed)

    # Test with a non-None preprocess_fn.
    def preprocess_fn(x):
      return x * 10.

    agent = MockDQNAgent(num_actions=5, network=MockNetwork,
                         preprocess_fn=preprocess_fn)
    self.assertEqual(5, agent.network_def.num_actions)
    self.assertTrue(agent.network_def.inputs_preprocessed)

  def testBeginEpisode(self):
    """Test the functionality of agent.begin_episode.

    Specifically, the action returned and its effect on state.
    """
    agent = self._create_test_agent()
    # We fill up the state with 9s. On calling agent.begin_episode the state
    # should be reset to all 0s.
    agent.state.fill(9)
    first_observation = onp.ones(self.observation_shape + (1,))
    self.assertEqual(agent.begin_episode(first_observation), 0)
    # When the all-1s observation is received, it will be placed at the end of
    # the state.
    expected_state = self.zero_state
    expected_state[..., -1] = onp.ones(self.observation_shape)
    self.assertTrue(onp.array_equal(agent.state, expected_state))
    self.assertTrue(onp.array_equal(agent._observation,
                                    first_observation[:, :, 0]))
    # No training happens in eval mode.
    self.assertEqual(agent.training_steps, 0)

    # This will now cause training to happen.
    agent.eval_mode = False
    # Having a low replay memory add_count will prevent any of the
    # train/prefetch/sync ops from being called.
    agent._replay.add_count = 0
    second_observation = onp.ones(self.observation_shape + (1,)) * 2
    agent.begin_episode(second_observation)
    # The agent's state will be reset, so we will only be left with the all-2s
    # observation.
    expected_state[:, :, -1] = onp.full(self.observation_shape, 2)
    self.assertTrue(onp.array_equal(agent.state, expected_state))
    self.assertTrue(onp.array_equal(agent._observation,
                                    second_observation[:, :, 0]))
    # training_steps is incremented since we set eval_mode to False.
    self.assertEqual(agent.training_steps, 1)

  def testStepEval(self):
    """Test the functionality of agent.step() in eval mode.

    Specifically, the action returned, and confirm no training is happening.
    """
    agent = self._create_test_agent()
    base_observation = onp.ones(self.observation_shape + (1,))
    # This will reset state and choose a first action.
    agent.begin_episode(base_observation)
    # We mock the replay buffer to verify how the agent interacts with it.
    agent._replay = test_utils.MockReplayBuffer(is_jax=True)

    expected_state = self.zero_state
    num_steps = 10
    for step in range(1, num_steps + 1):
      # We make observation a multiple of step for testing purposes (to
      # uniquely identify each observation).
      observation = base_observation * step
      self.assertEqual(agent.step(reward=1, observation=observation), 0)
      stack_pos = step - num_steps - 1
      if stack_pos >= -self.stack_size:
        expected_state[:, :, stack_pos] = onp.full(self.observation_shape, step)
    self.assertTrue(onp.array_equal(agent.state, expected_state))
    self.assertTrue(onp.array_equal(
        agent._last_observation,
        onp.ones(self.observation_shape) * (num_steps - 1)))
    self.assertTrue(onp.array_equal(agent._observation, observation[:, :, 0]))
    # No training happens in eval mode.
    self.assertEqual(agent.training_steps, 0)
    # No transitions are added in eval mode.
    self.assertEqual(agent._replay.add.call_count, 0)

  def testStepTrain(self):
    """Test the functionality of agent.step() in train mode.

    Specifically, the action returned, and confirm training is happening.
    """
    agent = self._create_test_agent()
    agent.eval_mode = False
    base_observation = onp.ones(self.observation_shape + (1,))
    # We mock the replay buffer to verify how the agent interacts with it.
    agent._replay = test_utils.MockReplayBuffer(is_jax=True)
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
        expected_state[:, :, stack_pos] = onp.full(self.observation_shape, step)
      self.assertEqual(agent._replay.add.call_count, step)
      mock_args, _ = agent._replay.add.call_args
      self.assertTrue(onp.array_equal(last_observation[:, :, 0], mock_args[0]))
      self.assertEqual(0, mock_args[1])  # Action selected.
      self.assertEqual(1, mock_args[2])  # Reward received.
      self.assertFalse(mock_args[3])  # is_terminal
    self.assertTrue(onp.array_equal(agent.state, expected_state))
    self.assertTrue(onp.array_equal(
        agent._last_observation,
        onp.full(self.observation_shape, num_steps - 1)))
    self.assertTrue(onp.array_equal(agent._observation, observation[:, :, 0]))
    # We expect one more than num_steps because of the call to begin_episode.
    self.assertEqual(agent.training_steps, num_steps + 1)
    self.assertEqual(agent._replay.add.call_count, num_steps)

    agent.end_episode(reward=1)
    self.assertEqual(agent._replay.add.call_count, num_steps + 1)
    mock_args, _ = agent._replay.add.call_args
    self.assertTrue(onp.array_equal(observation[:, :, 0], mock_args[0]))
    self.assertEqual(0, mock_args[1])  # Action selected.
    self.assertEqual(1, mock_args[2])  # Reward received.
    self.assertTrue(mock_args[3])  # is_terminal

  def testNonTupleObservationShape(self):
    with self.assertRaises(AssertionError):
      self.observation_shape = 84
      _ = self._create_test_agent()

  def _custom_shapes_test(self, shape, dtype, stack_size):
    self.observation_shape = shape
    self.observation_dtype = dtype
    self.stack_size = stack_size
    self.zero_state = onp.zeros(shape + (stack_size,))
    agent = self._create_test_agent()
    agent.eval_mode = False
    base_observation = onp.ones(self.observation_shape + (1,))
    # We mock the replay buffer to verify how the agent interacts with it.
    agent._replay = test_utils.MockReplayBuffer(is_jax=True)
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
        expected_state[..., stack_pos] = onp.full(self.observation_shape, step)
      self.assertEqual(agent._replay.add.call_count, step)
      mock_args, _ = agent._replay.add.call_args
      self.assertTrue(onp.array_equal(last_observation[..., 0], mock_args[0]))
      self.assertEqual(0, mock_args[1])  # Action selected.
      self.assertEqual(1, mock_args[2])  # Reward received.
      self.assertFalse(mock_args[3])  # is_terminal
    self.assertTrue(onp.array_equal(agent.state, expected_state))
    self.assertTrue(onp.array_equal(
        agent._last_observation,
        onp.full(self.observation_shape, num_steps - 1)))
    self.assertTrue(onp.array_equal(agent._observation, observation[..., 0]))
    # We expect one more than num_steps because of the call to begin_episode.
    self.assertEqual(agent.training_steps, num_steps + 1)
    self.assertEqual(agent._replay.add.call_count, num_steps)

    agent.end_episode(reward=1)
    self.assertEqual(agent._replay.add.call_count, num_steps + 1)
    mock_args, _ = agent._replay.add.call_args
    self.assertTrue(onp.array_equal(observation[..., 0], mock_args[0]))
    self.assertEqual(0, mock_args[1])  # Action selected.
    self.assertEqual(1, mock_args[2])  # Reward received.
    self.assertTrue(mock_args[3])  # is_terminal

  def testStepTrainCustomObservationShapes(self):
    custom_shapes = [(1,), (4, 4), (6, 1), (1, 6), (1, 1, 6), (6, 6, 6, 6)]
    for shape in custom_shapes:
      self._custom_shapes_test(shape, jnp.uint8, 1)

  def testStepTrainCustomTypes(self):
    custom_types = [jnp.float32, jnp.uint8, jnp.int64]
    for dtype in custom_types:
      self._custom_shapes_test((4, 4), dtype, 1)

  def testStepTrainCustomStackSizes(self):
    custom_stack_sizes = [1, 4, 8]
    for stack_size in custom_stack_sizes:
      self._custom_shapes_test((4, 4), jnp.uint8, stack_size)

  def testBundlingWithNonexistentDirectory(self):
    agent = self._create_test_agent()
    self.assertIsNone(agent.bundle_and_checkpoint('/does/not/exist', 1))

  def testUnbundlingWithFailingReplayBuffer(self):
    agent = self._create_test_agent()
    bundle = {}
    # The ReplayBuffer will throw an exception since it is not able to load
    # the expected files, which will cause the unbundle() method to return
    # False.
    self.assertFalse(agent.unbundle(self._test_subdir, 1729, bundle))

  def testUnbundlingWithNoBundleDictionary(self):
    agent = self._create_test_agent()
    agent._replay = mock.Mock()
    self.assertFalse(agent.unbundle(self._test_subdir, 1729, None))

  def testPartialUnbundling(self):
    agent = self._create_test_agent(allow_partial_reload=True)
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
    agent = self._create_test_agent()
    # These values don't reflect the actual types of these attributes, but are
    # used merely for facility of testing.
    agent.state = 'state'
    agent._replay = mock.Mock()
    agent.training_steps = 'training_steps'
    iteration_number = 1729
    bundle = agent.bundle_and_checkpoint(self._test_subdir, iteration_number)
    keys = ['state', 'training_steps', 'online_params', 'target_params']
    for key in keys:
      self.assertIn(key, bundle)
      if 'params' not in key:
        self.assertEqual(key, bundle[key])

    agent.unbundle(self._test_subdir, iteration_number, bundle)
    for key in keys:
      if 'params' not in key:
        self.assertEqual(key, agent.__dict__[key])

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
      onp.testing.assert_almost_equal(
          dqn_agent.linearly_decaying_epsilon(decay_period,
                                              step,
                                              warmup_steps,
                                              epsilon),
          expected_epsilon, 0.01)


if __name__ == '__main__':
  absltest.main()

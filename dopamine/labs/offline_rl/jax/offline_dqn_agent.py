# coding=utf-8
# Copyright 2023 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compact implementation of an offline DQN agent in JAX."""

from absl import logging
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.labs.offline_rl import fixed_replay
from dopamine.labs.offline_rl.jax import networks  # pylint: disable=unused-import
from dopamine.labs.offline_rl.rlu_tfds import tfds_replay
import gin
import numpy as onp


@gin.configurable
class OfflineJaxDQNAgent(dqn_agent.JaxDQNAgent):
  """A JAX implementation of the Offline DQN agent."""

  def __init__(self,
               num_actions,
               replay_data_dir,
               summary_writer=None,
               replay_buffer_builder=None,
               use_tfds=False):
    """Initializes the agent and constructs the necessary components.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      replay_data_dir: str, log Directory from which to load the replay buffer.
      summary_writer: SummaryWriter object for outputting training statistics
      replay_buffer_builder: Callable object that takes "self" as an argument
        and returns a replay buffer to use for training offline. If None, it
        will use the default FixedReplayBuffer.
      use_tfds: Whether to use tfds replay buffer.
    """
    logging.info('Creating %s agent with the following parameters:',
                 self.__class__.__name__)
    logging.info('\t replay directory: %s', replay_data_dir)
    self.replay_data_dir = replay_data_dir
    self._use_tfds = use_tfds
    if replay_buffer_builder is not None:
      self._build_replay_buffer = replay_buffer_builder

    # update_period=1 is a sane default for offline RL. However, this
    # can still be overridden with gin config.
    super().__init__(
        num_actions, update_period=1, summary_writer=summary_writer)

  def _build_replay_buffer(self):
    """Creates the fixed replay buffer used by the agent."""

    if not self._use_tfds:
      return fixed_replay.JaxFixedReplayBuffer(
          data_dir=self.replay_data_dir,
          observation_shape=self.observation_shape,
          stack_size=self.stack_size,
          update_horizon=self.update_horizon,
          gamma=self.gamma,
          observation_dtype=self.observation_dtype)

    dataset_name = tfds_replay.get_atari_ds_name_from_replay(
        self.replay_data_dir)
    return tfds_replay.JaxFixedReplayBufferTFDS(
        replay_capacity=gin.query_parameter(
            'JaxFixedReplayBuffer.replay_capacity'),
        batch_size=gin.query_parameter('JaxFixedReplayBuffer.batch_size'),
        dataset_name=dataset_name,
        stack_size=self.stack_size,
        update_horizon=self.update_horizon,
        gamma=self.gamma)

  def _sample_from_replay_buffer(self):
    if self._use_tfds:
      self.replay_elements = self._replay.sample_transition_batch()
    else:
      super()._sample_from_replay_buffer()

  def reload_data(self):
    # This doesn't do anything for tfds replay.
    self._replay.reload_data()

  def step(self, reward, observation):
    """Returns the agent's next action and update agent's state.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    self._record_observation(observation)
    self._rng, self.action = dqn_agent.select_action(
        self.network_def, self.online_params, self.state, self._rng,
        self.num_actions, self.eval_mode, self.epsilon_eval, self.epsilon_train,
        self.epsilon_decay_period, self.training_steps, self.min_replay_history,
        self.epsilon_fn)
    self.action = onp.asarray(self.action)
    return self.action

  def train_step(self):
    """Exposes the train step for offline learning."""
    super()._train_step()

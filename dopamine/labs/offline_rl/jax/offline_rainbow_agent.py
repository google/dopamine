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
"""Compact implementation of an offline BC + Rainbow agent in JAX."""

import functools

from absl import logging
from dopamine.jax import losses
from dopamine.jax.agents.full_rainbow import full_rainbow_agent
from dopamine.jax.agents.rainbow import rainbow_agent
from dopamine.labs.offline_rl import fixed_replay
from dopamine.labs.offline_rl.rlu_tfds import tfds_replay
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tensorflow as tf

c51_loss_fn = jax.vmap(losses.softmax_cross_entropy_loss_with_logits)
huber_loss_fn = jax.vmap(losses.huber_loss)


@functools.partial(jax.vmap, in_axes=(None, 0, None))
def get_logits_and_q_values(model, states, rng):
  outputs = model(states, key=rng)
  return (jnp.squeeze(outputs.logits), jnp.squeeze(outputs.q_values))


@functools.partial(
    jax.jit,
    static_argnames=('network_def', 'optimizer', 'double_dqn', 'distributional',
                     'cumulative_gamma', 'bc_coefficient', 'td_coefficient'))
def train(
    network_def, online_params, target_params, optimizer, optimizer_state,
    states, actions, next_states, rewards, terminals, support, cumulative_gamma,
    double_dqn, distributional, rng, bc_coefficient=0.0, td_coefficient=1.0):
  """Run a training step."""

  # Split the current rng into 2 for updating the rng after this call
  rng, rng1, rng2 = jax.random.split(rng, num=3)

  def q_online(state, key):
    return network_def.apply(online_params, state, key=key, support=support)

  def q_target(state, key):
    return network_def.apply(target_params, state, key=key, support=support)

  def loss_fn(params, target):
    """Computes the distributional loss for C51 or huber loss for DQN."""

    def q_func(state, key):
      return network_def.apply(params, state, key=key, support=support)

    if distributional:
      logits, q_values = get_logits_and_q_values(q_func, states, rng)
      # Fetch the logits for its selected action. We use vmap to perform this
      # indexing across the batch.
      chosen_action_logits = jax.vmap(lambda x, y: x[y])(logits, actions)
      mean_td_loss = jnp.mean(c51_loss_fn(target, chosen_action_logits))
      replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
    else:
      q_values = jnp.squeeze(
          full_rainbow_agent.get_q_values(q_func, states, rng))
      replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
      mean_td_loss = jnp.mean(huber_loss_fn(target, replay_chosen_q))

    bc_loss = jnp.mean(
        jax.scipy.special.logsumexp(q_values, axis=-1) - replay_chosen_q)
    mean_loss = (td_coefficient * mean_td_loss + bc_coefficient * bc_loss)
    return mean_loss, (mean_td_loss, bc_loss)

  target = full_rainbow_agent.target_output(q_online, q_target, next_states,
                                            rewards, terminals, support,
                                            cumulative_gamma, double_dqn,
                                            distributional, rng1)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  # Get the unweighted loss without taking its mean for updating priorities.
  # outputs[1] correspond to the per-example TD loss.
  (loss, outputs), grad = grad_fn(online_params, target)
  updates, optimizer_state = optimizer.update(
      grad, optimizer_state, params=online_params)
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, loss, outputs, rng2


@gin.configurable
class OfflineJaxRainbowAgent(full_rainbow_agent.JaxFullRainbowAgent):
  """Offline Rainbow agent with BC regularization (akin to CQL)."""

  def __init__(self,
               num_actions,
               replay_data_dir,
               td_coefficient=1.0,
               bc_coefficient=0.0,
               summary_writer=None,
               add_return_to_go=False,
               use_tfds=True):
    """Initializes the agent and constructs the necessary components.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      replay_data_dir: str, log Directory from which to load the replay buffer.
      td_coefficient: float, Coefficient for the DR3 regularizer.
      bc_coefficient: float, Coefficient for the CQL loss.
      summary_writer: SummaryWriter object for outputting training statistics.
      add_return_to_go: Whether to add return to go to the replay data.
      use_tfds: Whether to use tfds replay buffer.
    """

    logging.info('Creating OfflineJaxRainbowAgent with the parameters:')
    logging.info('\t replay directory: %s', replay_data_dir)
    logging.info('\t TD coefficient: %s', td_coefficient)
    logging.info('\t CQL coefficient: %s', bc_coefficient)
    logging.info('\t Return To Go: %s', add_return_to_go)

    self.replay_data_dir = replay_data_dir
    self._use_tfds = use_tfds
    self._td_coefficient = td_coefficient
    self._bc_coefficient = bc_coefficient
    self._add_return_to_go = add_return_to_go

    super().__init__(
        num_actions,
        noisy=False,  # No need for noisy networks for offline RL.
        replay_scheme='uniform',  # Uniform replay is default for offline RL.
        summary_writer=summary_writer)

  def _training_step_update(self):
    """Runs a single training step."""
    self._sample_from_replay_buffer()
    states = self.preprocess_fn(self.replay_elements['state'])
    next_states = self.preprocess_fn(self.replay_elements['next_state'])

    (self.optimizer_state,
     self.online_params, mean_loss, aux_info, self._rng) = train(
         self.network_def, self.online_params, self.target_network_params,
         self.optimizer, self.optimizer_state, states,
         self.replay_elements['action'], next_states,
         self.replay_elements['reward'], self.replay_elements['terminal'],
         self._support, self.cumulative_gamma, self._double_dqn,
         self._distributional, self._rng,
         bc_coefficient=self._bc_coefficient,
         td_coefficient=self._td_coefficient)

    td_loss, bc_loss = aux_info
    if (self.training_steps > 0 and
        self.training_steps % self.summary_writing_frequency == 0):
      if self.summary_writer is not None:
        with self.summary_writer.as_default():
          tf.summary.scalar('Losses/Aggregate', mean_loss,
                            step=self.training_steps)
          tf.summary.scalar('Losses/TD', td_loss,
                            step=self.training_steps)
          tf.summary.scalar('Losses/BCLoss', bc_loss,
                            step=self.training_steps)
        self.summary_writer.flush()
    if self._use_tfds:
      self.log_gradient_steps_per_epoch()

    if self.training_steps % self.target_update_period == 0:
      self._sync_weights()
    self.training_steps += 1

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
        gamma=self.gamma,
        return_to_go=self._add_return_to_go)

  def log_gradient_steps_per_epoch(self):
    num_steps_per_epoch = self._replay.gradient_steps_per_epoch
    steps_per_epoch = self.training_steps / num_steps_per_epoch
    if self.summary_writer is not None:
      with self.summary_writer.as_default():
        tf.summary.scalar(
            'Info/EpochFractionSteps',
            steps_per_epoch,
            step=self.training_steps)

  def _sample_from_replay_buffer(self):
    if self._use_tfds:
      self.replay_elements = self._replay.sample_transition_batch()
    else:
      super()._sample_from_replay_buffer()

  def reload_data(self):
    # This doesn't do anything for tfds replay.
    self._replay.reload_data()

  def train_step(self):
    super()._train_step()

  def step(self, reward, observation):
    """Returns the agent's next action and update agent's state.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    self._record_observation(observation)
    self._rng, self.action = rainbow_agent.select_action(
        self.network_def, self.online_params, self.state, self._rng,
        self.num_actions, self.eval_mode, self.epsilon_eval, self.epsilon_train,
        self.epsilon_decay_period, self.training_steps, self.min_replay_history,
        self.epsilon_fn, self._support)
    self.action = onp.asarray(self.action)
    return self.action

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
"""Compact implementation of an offline DR3 + DQN agent in JAX."""

import functools

from absl import logging
from dopamine.jax import losses
from dopamine.jax.agents.dqn import dqn_agent as base_dqn_agent
from dopamine.labs.offline_rl.jax import offline_dqn_agent
from dopamine.metrics import statistics_instance
import gin
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf


@functools.partial(jax.jit, static_argnums=(0, 3, 10, 11, 12))
def train(network_def, online_params, target_params, optimizer, optimizer_state,
          states, actions, next_states, rewards, terminals, cumulative_gamma,
          dr3_coefficient, cql_coefficient):
  """Run the training step."""
  def loss_fn(params, bellman_target):
    def q_online(state):
      return network_def.apply(params, state)

    model_output = jax.vmap(q_online)(states)
    representations = jnp.squeeze(model_output.representation)
    next_states_model_output = jax.vmap(q_online)(next_states)
    next_state_representations = jnp.squeeze(
        next_states_model_output.representation)
    dr3_loss = compute_dr3_loss(representations, next_state_representations)

    # Q-learning loss
    q_values = jnp.squeeze(model_output.q_values)
    replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
    bellman_loss = jnp.mean(
        jax.vmap(losses.huber_loss)(bellman_target, replay_chosen_q))

    # CQL Loss
    cql_loss = jnp.mean(
        jax.scipy.special.logsumexp(q_values, axis=-1) - replay_chosen_q)
    loss = (
        bellman_loss + dr3_coefficient * dr3_loss + cql_coefficient * cql_loss)

    q_vals_data = jnp.mean(replay_chosen_q)
    q_vals_pi = jnp.mean(jnp.max(q_values, axis=1))

    return jnp.mean(loss), (bellman_loss, dr3_loss, cql_loss, q_vals_data,
                            q_vals_pi)

  def q_target(state):
    return network_def.apply(target_params, state)

  # Making a temp function for measuring the bellman loss
  def q_online_temp(state):
    return network_def.apply(online_params, state)

  bellman_target = base_dqn_agent.target_q(q_target, next_states, rewards,
                                           terminals, cumulative_gamma)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, component_losses), grad = grad_fn(online_params, bellman_target)

  # Compute actual Bellman targets
  actual_bellman_target = base_dqn_agent.target_q(q_online_temp, next_states,
                                                  rewards, terminals,
                                                  cumulative_gamma)
  _, (actual_bellman_loss, _, _, _, _) = loss_fn(online_params,
                                                 actual_bellman_target)
  bellman_loss, dr3_loss, cql_loss, q_vals_data, q_vals_pi = component_losses
  updates, optimizer_state = optimizer.update(grad, optimizer_state)
  online_params = optax.apply_updates(online_params, updates)
  return (optimizer_state, online_params, loss, bellman_loss, dr3_loss,
          cql_loss, actual_bellman_loss, q_vals_data, q_vals_pi)


def compute_dr3_loss(state_representations, next_state_representations):
  """Minimizes dot product between state and next state representations."""
  dot_products = jnp.einsum(
      'ij,ij->i', state_representations, next_state_representations)
  # Minimize |\phi(s) \phi(s')|
  return jnp.mean(jnp.abs(dot_products))


@gin.configurable
class OfflineJaxDR3Agent(offline_dqn_agent.OfflineJaxDQNAgent):
  """A JAX implementation of the Offline DQN agent with DR3 regularizer."""

  def __init__(self,
               num_actions,
               replay_data_dir,
               dr3_coefficient=0.0,
               cql_coefficient=0.0,
               summary_writer=None,
               replay_buffer_builder=None):
    """Initializes the agent and constructs the necessary components.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      replay_data_dir: str, log Directory from which to load the replay buffer.
      dr3_coefficient: float, Coefficient for the DR3 regularizer.
      cql_coefficient: float, Coefficient for the CQL loss.
      summary_writer: SummaryWriter object for outputting training statistics.
      replay_buffer_builder: Callable object that takes "self" as an argument
        and returns a replay buffer to use for training offline. If None,
        it will use the default FixedReplayBuffer.
    """
    logging.info('Creating %s agent with the following parameters:',
                 self.__class__.__name__)
    logging.info('\t dr3_coefficient: %s', dr3_coefficient)
    logging.info('\t CQL coefficient: %s', cql_coefficient)
    super().__init__(
        num_actions,
        replay_data_dir=replay_data_dir,
        replay_buffer_builder=replay_buffer_builder,
        summary_writer=summary_writer)
    self._dr3_coefficient = dr3_coefficient
    self._cql_coefficient = cql_coefficient

  def train_step(self):
    """Runs a single training step."""
    if self.training_steps % self.update_period == 0:
      self._sample_from_replay_buffer()
      (self.optimizer_state, self.online_params, loss, bellman_loss, dr3_loss,
       cql_loss, actual_bellman_loss, q_data, q_pi) = train(
           self.network_def, self.online_params, self.target_network_params,
           self.optimizer, self.optimizer_state,
           self.preprocess_fn(self.replay_elements['state']),
           self.replay_elements['action'],
           self.preprocess_fn(self.replay_elements['next_state']),
           self.replay_elements['reward'],
           self.replay_elements['terminal'],
           self.cumulative_gamma,
           self._dr3_coefficient,
           self._cql_coefficient)

      if (self.training_steps > 0 and
          self.training_steps % self.summary_writing_frequency == 0):
        if self.summary_writer is not None:
          with self.summary_writer.as_default():
            tf.summary.scalar('Losses/Aggregate', loss,
                              step=self.training_steps)
            tf.summary.scalar('Losses/Bellman', bellman_loss,
                              step=self.training_steps)
            tf.summary.scalar('Losses/DR3', dr3_loss,
                              step=self.training_steps)
            tf.summary.scalar('Losses/CQLLoss', cql_loss,
                              step=self.training_steps)
            tf.summary.scalar('Losses/BellmanActual', actual_bellman_loss,
                              step=self.training_steps)
            tf.summary.scalar('Losses/q_vals_data', q_data,
                              step=self.training_steps)
            tf.summary.scalar('Losses/q_vals_pi', q_pi,
                              step=self.training_steps)
          self.summary_writer.flush()
        if hasattr(self, 'collector_dispatcher'):
          self.collector_dispatcher.write(
              [
                  statistics_instance.StatisticsInstance(
                      'Loss', np.asarray(loss), step=self.training_steps),
              ],
              collector_allowlist=self._collector_allowlist)
    if self.training_steps % self.target_update_period == 0:
      self._sync_weights()
    self.training_steps += 1

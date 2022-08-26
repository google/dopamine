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
"""An extension of Rainbow to perform quantile regression.

This loss is computed as in "Distributional Reinforcement Learning with Quantile
Regression" - Dabney et. al, 2017"
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from dopamine.jax import networks
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.metrics import statistics_instance
from dopamine.replay_memory import prioritized_replay_buffer
import gin
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf


@functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, None))
def target_distribution(target_network, next_states, rewards, terminals,
                        cumulative_gamma):
  """Builds the Quantile target distribution as per Dabney et al. (2017).

  Args:
    target_network: Jax Module used for the target network.
    next_states: numpy array of batched next states.
    rewards: numpy array of batched rewards.
    terminals: numpy array of batched terminals.
    cumulative_gamma: float, cumulative gamma to use.

  Returns:
    The target distribution from the replay.
  """
  is_terminal_multiplier = 1. - terminals.astype(jnp.float32)
  # Incorporate terminal state to discount factor.
  gamma_with_terminal = cumulative_gamma * is_terminal_multiplier
  next_state_target_outputs = target_network(next_states)
  q_values = jnp.squeeze(next_state_target_outputs.q_values)
  next_qt_argmax = jnp.argmax(q_values)
  logits = jnp.squeeze(next_state_target_outputs.logits)
  next_logits = logits[next_qt_argmax]
  return jax.lax.stop_gradient(rewards + gamma_with_terminal * next_logits)


@functools.partial(jax.jit, static_argnums=(0, 3, 10, 11, 12))
def train(network_def, online_params, target_params, optimizer, optimizer_state,
          states, actions, next_states, rewards, terminals, kappa, num_atoms,
          cumulative_gamma):
  """Run a training step."""
  def loss_fn(params, target):
    def q_online(state):
      return network_def.apply(params, state)

    logits = jax.vmap(q_online)(states).logits
    logits = jnp.squeeze(logits)
    # Fetch the logits for its selected action. We use vmap to perform this
    # indexing across the batch.
    chosen_action_logits = jax.vmap(lambda x, y: x[y])(logits, actions)
    bellman_errors = (target[:, None, :] -
                      chosen_action_logits[:, :, None])  # Input `u' of Eq. 9.
    # Eq. 9 of paper.
    huber_loss = (
        (jnp.abs(bellman_errors) <= kappa).astype(jnp.float32) *
        0.5 * bellman_errors ** 2 +
        (jnp.abs(bellman_errors) > kappa).astype(jnp.float32) *
        kappa * (jnp.abs(bellman_errors) - 0.5 * kappa))

    tau_hat = ((jnp.arange(num_atoms, dtype=jnp.float32) + 0.5) /
               num_atoms)  # Quantile midpoints.  See Lemma 2 of paper.
    # Eq. 10 of paper.
    tau_bellman_diff = jnp.abs(
        tau_hat[None, :, None] - (bellman_errors < 0).astype(jnp.float32))
    quantile_huber_loss = tau_bellman_diff * huber_loss
    # Sum over tau dimension, average over target value dimension.
    loss = jnp.sum(jnp.mean(quantile_huber_loss, 2), 1)
    return jnp.mean(loss), loss

  def q_target(state):
    return network_def.apply(target_params, state)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  target = target_distribution(q_target,
                               next_states,
                               rewards,
                               terminals,
                               cumulative_gamma)
  (mean_loss, loss), grad = grad_fn(online_params, target)
  updates, optimizer_state = optimizer.update(grad, optimizer_state,
                                              params=online_params)
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, loss, mean_loss


@gin.configurable
class JaxQuantileAgent(dqn_agent.JaxDQNAgent):
  """An implementation of Quantile regression DQN agent."""

  def __init__(self,
               num_actions,
               observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
               stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
               network=networks.QuantileNetwork,
               kappa=1.0,
               num_atoms=200,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=50000,
               update_period=4,
               target_update_period=10000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.1,
               epsilon_eval=0.05,
               epsilon_decay_period=1000000,
               replay_scheme='prioritized',
               optimizer='adam',
               summary_writer=None,
               summary_writing_frequency=500,
               seed=None,
               allow_partial_reload=False):
    """Initializes the agent and constructs the Graph.

    Args:
      num_actions: Int, number of actions the agent can take at any state.
      observation_shape: tuple of ints or an int. If single int, the observation
        is assumed to be a 2D square.
      observation_dtype: DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to jnp.float32.
      stack_size: int, number of frames to use in state stack.
      network: flax.linen Module, expects 3 parameters: num_actions, num_atoms,
        network_type.
      kappa: Float, Huber loss cutoff.
      num_atoms: Int, the number of buckets for the value function distribution.
      gamma: Float, exponential decay factor as commonly used in the RL
        literature.
      update_horizon: Int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: Int, number of stored transitions for training to
        start.
      update_period: Int, period between DQN updates.
      target_update_period: Int, ppdate period for the target network.
      epsilon_fn: Function expecting 4 parameters: (decay_period, step,
        warmup_steps, epsilon), and which returns the epsilon value used for
        exploration during training.
      epsilon_train: Float, final epsilon for training.
      epsilon_eval: Float, epsilon during evaluation.
      epsilon_decay_period: Int, number of steps for epsilon to decay.
      replay_scheme: String, replay memory scheme to be used. Choices are:
        uniform - Standard (DQN) replay buffer (Mnih et al., 2015)
        prioritized - Prioritized replay buffer (Schaul et al., 2015)
      optimizer: str, name of optimizer to use.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      seed: int, a seed for DQN's internal RNG, used for initialization and
        sampling actions. If None, will use the current time in nanoseconds.
      allow_partial_reload: bool, whether we allow reloading a partial agent
        (for instance, only the network parameters).
    """
    self._num_atoms = num_atoms
    self._kappa = kappa
    self._replay_scheme = replay_scheme

    super(JaxQuantileAgent, self).__init__(
        num_actions=num_actions,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        stack_size=stack_size,
        network=functools.partial(network, num_atoms=num_atoms),
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        optimizer=optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency,
        seed=seed,
        allow_partial_reload=allow_partial_reload)

  def _build_networks_and_optimizer(self):
    self._rng, rng = jax.random.split(self._rng)
    self.online_params = self.network_def.init(rng, x=self.state)
    self.optimizer = dqn_agent.create_optimizer(self._optimizer_name)
    self.optimizer_state = self.optimizer.init(self.online_params)
    self.target_network_params = self.online_params

  def _build_replay_buffer(self):
    """Creates the replay buffer used by the agent."""
    if self._replay_scheme not in ['uniform', 'prioritized']:
      raise ValueError('Invalid replay scheme: {}'.format(self._replay_scheme))
    # Both replay schemes use the same data structure, but the 'uniform' scheme
    # sets all priorities to the same value (which yields uniform sampling).
    return prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype)

  def _train_step(self):
    """Runs a single training step.

    Runs training if both:
      (1) A minimum number of frames have been added to the replay buffer.
      (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online_params to target_network_params if training
    steps is a multiple of target update period.
    """
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()
        self.optimizer_state, self.online_params, loss, mean_loss = train(
            self.network_def,
            self.online_params,
            self.target_network_params,
            self.optimizer,
            self.optimizer_state,
            self.preprocess_fn(self.replay_elements['state']),
            self.replay_elements['action'],
            self.preprocess_fn(self.replay_elements['next_state']),
            self.replay_elements['reward'],
            self.replay_elements['terminal'],
            self._kappa,
            self._num_atoms,
            self.cumulative_gamma)
        if self._replay_scheme == 'prioritized':
          # The original prioritized experience replay uses a linear exponent
          # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of
          # 0.5 on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders)
          # suggested a fixed exponent actually performs better, except on Pong.
          probs = self.replay_elements['sampling_probabilities']
          loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
          loss_weights /= jnp.max(loss_weights)

          # Rainbow and prioritized replay are parametrized by an exponent
          # alpha, but in both cases it is set to 0.5 - for simplicity's sake we
          # leave it as is here, using the more direct sqrt(). Taking the square
          # root "makes sense", as we are dealing with a squared loss.  Add a
          # small nonzero value to the loss to avoid 0 priority items. While
          # technically this may be okay, setting all items to 0 priority will
          # cause troubles, and also result in 1.0 / 0.0 = NaN correction terms.
          self._replay.set_priority(self.replay_elements['indices'],
                                    jnp.sqrt(loss + 1e-10))

          # Weight the loss by the inverse priorities.
          loss = loss_weights * loss
          mean_loss = jnp.mean(loss)
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          with self.summary_writer.as_default():
            tf.summary.scalar('QuantileLoss', mean_loss,
                              step=self.training_steps)
          self.summary_writer.flush()
          if hasattr(self, 'collector_dispatcher'):
            self.collector_dispatcher.write(
                [statistics_instance.StatisticsInstance(
                    'Loss', np.asarray(mean_loss), step=self.training_steps),
                 ],
                collector_allowlist=self._collector_allowlist)
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1

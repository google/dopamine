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
"""The implicit quantile networks (IQN) agent.

The agent follows the description given in "Implicit Quantile Networks for
Distributional RL" (Dabney et. al, 2018).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools



from dopamine.jax import networks
from dopamine.jax.agents.dqn import dqn_agent
from flax import nn
import gin
import jax
import jax.numpy as jnp
import tensorflow as tf


@functools.partial(jax.jit, static_argnums=(3, 4, 5))
def train(online_network, target_quantile_values, replay_elements,
          num_tau_samples, num_tau_prime_samples, kappa, rng, optimizer):
  """Run a training step."""
  def loss_fn(model):
    batch_size = replay_elements['reward'].shape[0]
    # Shape: (num_tau_samples x batch_size) x num_actions.
    model_output = model(replay_elements['state'],
                         num_quantiles=num_tau_samples,
                         rng=rng)
    quantile_values = model_output.quantile_values
    quantiles = model_output.quantiles
    indices = jnp.arange(num_tau_samples * batch_size)
    reshaped_actions = jnp.tile(replay_elements['action'], [num_tau_samples])
    chosen_action_quantile_values = quantile_values[indices, reshaped_actions]
    # Reshape to num_tau_samples x batch_size x 1 since this is the manner
    # in which the quantile values are tiled.
    chosen_action_quantile_values = jnp.reshape(chosen_action_quantile_values,
                                                [num_tau_samples,
                                                 batch_size, 1])
    # Transpose dimensions so that the dimensionality is batch_size x
    # num_tau_samples x 1 to prepare for computation of
    # Bellman errors.
    # Final shape of chosen_action_quantile_values:
    # batch_size x num_tau_samples x 1.
    chosen_action_quantile_values = jnp.transpose(
        chosen_action_quantile_values, [1, 0, 2])
    # Shape of bellman_erors and huber_loss:
    # batch_size x num_tau_prime_samples x num_tau_samples x 1.
    bellman_errors = (target_quantile_values[:, :, None, :] -
                      chosen_action_quantile_values[:, None, :, :])
    # The huber loss (see Section 2.3 of the paper) is defined via two cases:
    # case_one: |bellman_errors| <= kappa
    # case_two: |bellman_errors| > kappa
    huber_loss_case_one = (
        (jnp.abs(bellman_errors) <= kappa).astype(jnp.float32) *
        0.5 * bellman_errors ** 2)
    huber_loss_case_two = (
        (jnp.abs(bellman_errors) > kappa).astype(jnp.float32) *
        kappa * (jnp.abs(bellman_errors) - 0.5 * kappa))
    huber_loss = huber_loss_case_one + huber_loss_case_two
    # Reshape quantiles to batch_size x num_tau_samples x 1
    quantiles = jnp.reshape(quantiles, [num_tau_samples, batch_size, 1])
    quantiles = jnp.transpose(quantiles, [1, 0, 2])
    # Tile by num_tau_prime_samples along a new dimension. Shape is now
    # batch_size x num_tau_prime_samples x num_tau_samples x 1.
    # These quantiles will be used for computation of the quantile huber loss
    # below (see section 2.3 of the paper).
    quantiles = jnp.tile(
        quantiles[:, None, :, :],
        [1, num_tau_prime_samples, 1, 1]).astype(jnp.float32)
    # Shape: batch_size x num_tau_prime_samples x num_tau_samples x 1.
    quantile_huber_loss = (jnp.abs(quantiles - jax.lax.stop_gradient(
        (bellman_errors < 0).astype(jnp.float32))) * huber_loss) / kappa
    # Sum over current quantile value (num_tau_samples) dimension,
    # average over target quantile value (num_tau_prime_samples) dimension.
    # Shape: batch_size x num_tau_prime_samples x 1.
    loss = jnp.sum(quantile_huber_loss, axis=2)
    # Shape: batch_size x 1.
    loss = jnp.mean(loss, axis=1)
    return jnp.mean(loss)
  grad_fn = jax.value_and_grad(loss_fn)
  loss, grad = grad_fn(online_network)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss


@gin.configurable
class JaxImplicitQuantileAgent(dqn_agent.JaxDQNAgent):
  """An extension of Rainbow to perform implicit quantile regression."""

  def __init__(self,
               num_actions,
               observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
               stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
               network=networks.ImplicitQuantileNetwork,
               kappa=1.0,
               num_tau_samples=32,
               num_tau_prime_samples=32,
               num_quantile_samples=32,
               quantile_embedding_dim=64,
               double_dqn=False,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               replay_scheme='prioritized',
               optimizer='adam',
               summary_writer=None,
               summary_writing_frequency=500):
    """Initializes the agent and constructs the necessary components.

    Most of this constructor's parameters are IQN-specific hyperparameters whose
    values are taken from Dabney et al. (2018).

    Args:
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints or an int. If single int, the observation
        is assumed to be a 2D square.
      observation_dtype: DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to jnp.float32.
      stack_size: int, number of frames to use in state stack.
      network: flax.nn Module that is initialized by shape in _create_network
        below. See dopamine.jax.networks.JaxImplicitQuantileNetwork as an
        example.
      kappa: float, Huber loss cutoff.
      num_tau_samples: int, number of online quantile samples for loss
        estimation.
      num_tau_prime_samples: int, number of target quantile samples for loss
        estimation.
      num_quantile_samples: int, number of quantile samples for computing
        Q-values.
      quantile_embedding_dim: int, embedding dimension for the quantile input.
      double_dqn: boolean, whether to perform double DQN style learning
        as described in Van Hasselt et al.: https://arxiv.org/abs/1509.06461.
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters:
        (decay_period, step, warmup_steps, epsilon). This function should return
        the epsilon value used for exploration during training.
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      replay_scheme: str, 'prioritized' or 'uniform', the sampling scheme of the
        replay memory.
      optimizer: str, name of optimizer to use.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
    """
    self.kappa = kappa
    # num_tau_samples = N below equation (3) in the paper.
    self.num_tau_samples = num_tau_samples
    # num_tau_prime_samples = N' below equation (3) in the paper.
    self.num_tau_prime_samples = num_tau_prime_samples
    # num_quantile_samples = k below equation (3) in the paper.
    self.num_quantile_samples = num_quantile_samples
    # quantile_embedding_dim = n above equation (4) in the paper.
    self.quantile_embedding_dim = quantile_embedding_dim
    # option to perform double dqn.
    self.double_dqn = double_dqn

    super(JaxImplicitQuantileAgent, self).__init__(
        num_actions=num_actions,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        stack_size=stack_size,
        network=network.partial(quantile_embedding_dim=quantile_embedding_dim),
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
        summary_writing_frequency=summary_writing_frequency)

  def _create_network(self, name):
    r"""Builds an Implicit Quantile ConvNet.

    Args:
      name: str, this name is passed to the Jax Module.
    Returns:
      network: Jax Model, the network instantiated by Jax.
    """
    _, initial_params = self.network.init_by_shape(
        self._rng, name=name,
        input_specs=[(self.state.shape, self.observation_dtype)],
        num_quantiles=self.num_tau_samples,
        rng=self._rng_input())
    return nn.Model(self.network, initial_params)

  def _select_action(self):
    """Select an action from the set of available actions.

    Chooses an action randomly with probability self._calculate_epsilon(), and
    otherwise acts greedily according to the current Q-value estimates.

    Returns:
       int, the selected action.
    """
    if self.eval_mode:
      epsilon = self.epsilon_eval
    else:
      epsilon = self.epsilon_fn(
          self.epsilon_decay_period,
          self.training_steps,
          self.min_replay_history,
          self.epsilon_train)
    if jax.random.uniform(self._rng_input()) <= epsilon:
      # Choose a random action with probability epsilon.
      return jax.random.randint(self._rng_input(), (), 0, self.num_actions)
    else:
      # Choose the action with highest Q-value at the current state.
      q_values = jnp.mean(
          self.online_network(self.state,
                              num_quantiles=self.num_quantile_samples,
                              rng=self._rng_input()).quantile_values, axis=0)
      return jnp.argmax(q_values, axis=0)

  def _build_target_quantile_values(self):
    """Build the target for return values at given quantiles.

    Returns:
      The target quantile values.
    """
    batch_size = self.replay_elements['reward'].shape[0]
    # Shape of rewards: (num_tau_prime_samples x batch_size).
    rewards = self.replay_elements['reward'][:, None]
    rewards = jnp.tile(rewards, [self.num_tau_prime_samples, 1])
    rewards = jnp.squeeze(rewards)
    is_terminal_multiplier = (
        1. - self.replay_elements['terminal'].astype(jnp.float32))
    # Incorporate terminal state to discount factor.
    # size of gamma_with_terminal: (num_tau_prime_samples x batch_size).
    gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
    gamma_with_terminal = jnp.tile(gamma_with_terminal[:, None],
                                   [self.num_tau_prime_samples, 1])
    gamma_with_terminal = jnp.squeeze(gamma_with_terminal)
    # Compute Q-values which are used for action selection for the next states
    # in the replay buffer. Compute the argmax over the Q-values.
    if self.double_dqn:
      outputs_action = self.online_network(
          self.replay_elements['next_state'],
          num_quantiles=self.num_quantile_samples,
          rng=self._rng_input())
    else:
      outputs_action = self.target_network(
          self.replay_elements['next_state'],
          num_quantiles=self.num_quantile_samples,
          rng=self._rng_input())
    # Shape: (num_quantile_samples x batch_size) x num_actions.
    target_quantile_values_action = outputs_action.quantile_values
    # Shape: num_quantile_samples x batch_size x num_actions.
    target_quantile_values_action = jnp.reshape(target_quantile_values_action,
                                                [self.num_quantile_samples,
                                                 batch_size,
                                                 self.num_actions])
    target_q_values = jnp.squeeze(
        jnp.mean(target_quantile_values_action, axis=0))
    # Shape: batch_size.
    next_qt_argmax = jnp.argmax(target_q_values, axis=1)
    # Get the indices of the maximium Q-value across the action dimension.
    # Shape of next_qt_argmax: (num_tau_prime_samples x batch_size).
    next_state_target_outputs = self.target_network(
        self.replay_elements['next_state'],
        num_quantiles=self.num_tau_prime_samples,
        rng=self._rng_input())
    next_qt_argmax = jnp.tile(
        next_qt_argmax[:, None], [self.num_tau_prime_samples, 1])
    next_qt_argmax = jnp.squeeze(next_qt_argmax)
    batch_indices = jnp.arange(self.num_tau_prime_samples * batch_size)
    target_quantile_values = next_state_target_outputs.quantile_values[
        batch_indices, next_qt_argmax]
    target_quantile_values = (
        rewards + gamma_with_terminal * target_quantile_values)
    # Reshape to self.num_tau_prime_samples x batch_size x 1 since this is
    # the manner in which the target_quantile_values are tiled.
    target_quantile_values = jnp.reshape(target_quantile_values,
                                         [self.num_tau_prime_samples,
                                          batch_size, 1])
    # Transpose dimensions so that the dimensionality is batch_size x
    # self.num_tau_prime_samples x 1 to prepare for computation of
    # Bellman errors.
    # Final shape of target_quantile_values:
    # batch_size x num_tau_prime_samples x 1.
    target_quantile_values = jnp.transpose(target_quantile_values, [1, 0, 2])
    return target_quantile_values

  def _train_step(self):
    """Runs a single training step.

    Runs training if both:
      (1) A minimum number of frames have been added to the replay buffer.
      (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online_network to target_network if training steps
    is a multiple of target update period.
    """
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()
        self.optimizer, loss = train(
            self.online_network,
            jax.lax.stop_gradient(self._build_target_quantile_values()),
            self.replay_elements,
            self.num_tau_samples,
            self.num_tau_prime_samples,
            self.kappa,
            self._rng_input(),
            self.optimizer)
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          summary = tf.compat.v1.Summary(value=[
              tf.compat.v1.Summary.Value(tag='QuantileLoss',
                                         simple_value=loss)])
          self.summary_writer.add_summary(summary, self.training_steps)
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1

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
from dopamine.metrics import statistics_instance
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tensorflow as tf


@functools.partial(
    jax.vmap,
    in_axes=(None, None, None, 0, 0, 0, None, None, None, None, 0),
    out_axes=(0, 0))
def target_quantile_values(network_def, online_params, target_params,
                           next_states, rewards, terminals,
                           num_tau_prime_samples, num_quantile_samples,
                           cumulative_gamma, double_dqn, rng):
  """Build the target for return values at given quantiles.

  Args:
    network_def: Linen Module used for inference.
    online_params: Parameters used for the online network.
    target_params: Parameters used for the target network.
    next_states: numpy array of batched next states.
    rewards: numpy array of batched rewards.
    terminals: numpy array of batched terminals.
    num_tau_prime_samples: int, number of tau' samples (static_argnum).
    num_quantile_samples: int, number of quantile samples (static_argnum).
    cumulative_gamma: float, cumulative gamma to use (static_argnum).
    double_dqn: bool, whether to use double DQN (static_argnum).
    rng: Jax random number generator.

  Returns:
    Jax random number generator.
    The target quantile values.
  """
  rewards = jnp.tile(rewards, [num_tau_prime_samples])
  is_terminal_multiplier = 1. - terminals.astype(jnp.float32)
  # Incorporate terminal state to discount factor.
  gamma_with_terminal = cumulative_gamma * is_terminal_multiplier
  gamma_with_terminal = jnp.tile(gamma_with_terminal, [num_tau_prime_samples])
  rng, rng1, rng2 = jax.random.split(rng, num=3)
  # Compute Q-values which are used for action selection for the next states
  # in the replay buffer. Compute the argmax over the Q-values.
  if double_dqn:
    outputs_action = network_def.apply(online_params,
                                       next_states,
                                       num_quantiles=num_quantile_samples,
                                       rng=rng1)
  else:
    outputs_action = network_def.apply(target_params,
                                       next_states,
                                       num_quantiles=num_quantile_samples,
                                       rng=rng1)
  target_quantile_values_action = outputs_action.quantile_values
  target_q_values = jnp.squeeze(
      jnp.mean(target_quantile_values_action, axis=0))
  # Shape: batch_size.
  next_qt_argmax = jnp.argmax(target_q_values)
  # Get the indices of the maximium Q-value across the action dimension.
  # Shape of next_qt_argmax: (num_tau_prime_samples x batch_size).
  next_state_target_outputs = network_def.apply(
      target_params,
      next_states,
      num_quantiles=num_tau_prime_samples,
      rng=rng2)
  next_qt_argmax = jnp.tile(next_qt_argmax, [num_tau_prime_samples])
  target_quantile_vals = (
      jax.vmap(lambda x, y: x[y])(next_state_target_outputs.quantile_values,
                                  next_qt_argmax))
  target_quantile_vals = rewards + gamma_with_terminal * target_quantile_vals
  # We return with an extra dimension, which is expected by train.
  return rng, jax.lax.stop_gradient(target_quantile_vals[:, None])


@functools.partial(jax.jit, static_argnums=(0, 3, 10, 11, 12, 13, 14, 15))
def train(network_def, online_params, target_params, optimizer, optimizer_state,
          states, actions, next_states, rewards, terminals, num_tau_samples,
          num_tau_prime_samples, num_quantile_samples, cumulative_gamma,
          double_dqn, kappa, rng):
  """Run a training step."""
  batch_size = states.shape[0]
  def loss_fn(params, rng_input, target_quantile_vals):
    def online(state, key):
      return network_def.apply(params, state, num_quantiles=num_tau_samples,
                               rng=key)

    batched_rng = jnp.stack(jax.random.split(rng_input, num=batch_size))
    model_output = jax.vmap(online)(states, batched_rng)
    quantile_values = model_output.quantile_values
    quantiles = model_output.quantiles
    chosen_action_quantile_values = jax.vmap(lambda x, y: x[:, y][:, None])(
        quantile_values, actions)
    # Shape of bellman_erors and huber_loss:
    # batch_size x num_tau_prime_samples x num_tau_samples x 1.
    bellman_errors = (target_quantile_vals[:, :, None, :] -
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
    # Tile by num_tau_prime_samples along a new dimension. Shape is now
    # batch_size x num_tau_prime_samples x num_tau_samples x 1.
    # These quantiles will be used for computation of the quantile huber loss
    # below (see section 2.3 of the paper).
    quantiles = jnp.tile(quantiles[:, None, :, :],
                         [1, num_tau_prime_samples, 1, 1]).astype(jnp.float32)
    # Shape: batch_size x num_tau_prime_samples x num_tau_samples x 1.
    quantile_huber_loss = (jnp.abs(quantiles - jax.lax.stop_gradient(
        (bellman_errors < 0).astype(jnp.float32))) * huber_loss) / kappa
    # Sum over current quantile value (num_tau_samples) dimension,
    # average over target quantile value (num_tau_prime_samples) dimension.
    # Shape: batch_size x num_tau_prime_samples x 1.
    loss = jnp.sum(quantile_huber_loss, axis=2)
    loss = jnp.mean(loss, axis=1)
    return jnp.mean(loss)

  rng, target_rng = jax.random.split(rng, num=2)
  batched_target_rng = jnp.stack(jax.random.split(target_rng, num=batch_size))
  _, target_quantile_vals = target_quantile_values(
      network_def,
      online_params,
      target_params,
      next_states,
      rewards,
      terminals,
      num_tau_prime_samples,
      num_quantile_samples,
      cumulative_gamma,
      double_dqn,
      batched_target_rng)
  grad_fn = jax.value_and_grad(loss_fn)
  rng, rng_input = jax.random.split(rng)
  loss, grad = grad_fn(online_params, rng_input, target_quantile_vals)
  updates, optimizer_state = optimizer.update(grad, optimizer_state,
                                              params=online_params)
  online_params = optax.apply_updates(online_params, updates)
  return rng, optimizer_state, online_params, loss


@functools.partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8, 9, 11, 12))
def select_action(network_def, params, state, rng, num_quantile_samples,
                  num_actions, eval_mode, epsilon_eval, epsilon_train,
                  epsilon_decay_period, training_steps, min_replay_history,
                  epsilon_fn):
  """Select an action from the set of available actions.

  Chooses an action randomly with probability self._calculate_epsilon(), and
  otherwise acts greedily according to the current Q-value estimates.

  Args:
    network_def: Linen Module to use for inference.
    params: Linen params (frozen dict) to use for inference.
    state: input state to use for inference.
    rng: Jax random number generator.
    num_quantile_samples: int, number of quantile samples (static_argnum).
    num_actions: int, number of actions (static_argnum).
    eval_mode: bool, whether we are in eval mode (static_argnum).
    epsilon_eval: float, epsilon value to use in eval mode (static_argnum).
    epsilon_train: float, epsilon value to use in train mode (static_argnum).
    epsilon_decay_period: float, decay period for epsilon value for certain
      epsilon functions, such as linearly_decaying_epsilon, (static_argnum).
    training_steps: int, number of training steps so far.
    min_replay_history: int, minimum number of steps in replay buffer
      (static_argnum).
    epsilon_fn: function used to calculate epsilon value (static_argnum).

  Returns:
    Jax random number generator.
    int, the selected action.
  """
  epsilon = jnp.where(eval_mode,
                      epsilon_eval,
                      epsilon_fn(epsilon_decay_period,
                                 training_steps,
                                 min_replay_history,
                                 epsilon_train))

  rng, rng1, rng2 = jax.random.split(rng, num=3)
  p = jax.random.uniform(rng1)
  return rng, jnp.where(p <= epsilon,
                        jax.random.randint(rng2, (), 0, num_actions),
                        jnp.argmax(jnp.mean(
                            network_def.apply(
                                params, state,
                                num_quantiles=num_quantile_samples,
                                rng=rng2).quantile_values, axis=0), axis=0))


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
               summary_writing_frequency=500,
               seed=None):
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
      network: flax.linen Module that is initialized by shape in _create_network
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
      seed: int, a seed for internal RNG, used for initialization and
        sampling actions. If None, will use the current time in nanoseconds.
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
        network=functools.partial(
            network, quantile_embedding_dim=quantile_embedding_dim),
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
        seed=seed)

  def _build_networks_and_optimizer(self):
    self._rng, rng = jax.random.split(self._rng)
    self.online_params = self.network_def.init(
        rng, x=self.state, num_quantiles=self.num_tau_samples,
        rng=self._rng)
    self.optimizer = dqn_agent.create_optimizer(self._optimizer_name)
    self.optimizer_state = self.optimizer.init(self.online_params)
    self.target_network_params = self.online_params

  def begin_episode(self, observation):
    """Returns the agent's first action for this episode.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      int, the selected action.
    """
    self._reset_state()
    self._record_observation(observation)

    if not self.eval_mode:
      self._train_step()

    self._rng, self.action = select_action(self.network_def,
                                           self.online_params,
                                           self.state,
                                           self._rng,
                                           self.num_quantile_samples,
                                           self.num_actions,
                                           self.eval_mode,
                                           self.epsilon_eval,
                                           self.epsilon_train,
                                           self.epsilon_decay_period,
                                           self.training_steps,
                                           self.min_replay_history,
                                           self.epsilon_fn)
    self.action = onp.asarray(self.action)
    return self.action

  def step(self, reward, observation):
    """Records the most recent transition and returns the agent's next action.

    We store the observation of the last time step since we want to store it
    with the reward.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    self._last_observation = self._observation
    self._record_observation(observation)

    if not self.eval_mode:
      self._store_transition(self._last_observation, self.action, reward, False)
      self._train_step()

    self._rng, self.action = select_action(self.network_def,
                                           self.online_params,
                                           self.preprocess_fn(self.state),
                                           self._rng,
                                           self.num_quantile_samples,
                                           self.num_actions,
                                           self.eval_mode,
                                           self.epsilon_eval,
                                           self.epsilon_train,
                                           self.epsilon_decay_period,
                                           self.training_steps,
                                           self.min_replay_history,
                                           self.epsilon_fn)
    self.action = onp.asarray(self.action)
    return self.action

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
        self._rng, self.optimizer_state, self.online_params, loss = train(
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
            self.num_tau_samples,
            self.num_tau_prime_samples,
            self.num_quantile_samples,
            self.cumulative_gamma,
            self.double_dqn,
            self.kappa,
            self._rng)
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          with self.summary_writer.as_default():
            tf.summary.scalar('QuantileLoss', loss, step=self.training_steps)
          self.summary_writer.flush()
          if hasattr(self, 'collector_dispatcher'):
            self.collector_dispatcher.write(
                [statistics_instance.StatisticsInstance(
                    'Loss', onp.asarray(loss), step=self.training_steps),
                 ],
                collector_allowlist=self._collector_allowlist)
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1

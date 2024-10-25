# coding=utf-8
# Copyright 2021 The Dopamine Authors.
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
"""Compact implementation of a Soft Actor-Critic agent in JAX.

Based on agent described in
  "Soft Actor-Critic Algorithms and Applications"
  by Tuomas Haarnoja et al.
  https://arxiv.org/abs/1812.05905
"""

import functools
import math
import operator
import time
from typing import Any, Mapping, Tuple

from absl import logging
from dopamine.jax import continuous_networks
from dopamine.jax import losses
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.replay_memory import accumulator
from dopamine.jax.replay_memory import replay_buffer
from dopamine.jax.replay_memory import samplers
# pylint: disable=unused-import
# This enables (experimental) networks for SAC from pixels.
# Note, that the full name import is required to avoid a naming
# collision with the short name import (continuous_networks) above.
import dopamine.labs.sac_from_pixels.continuous_networks
# pylint: enable=unused-import
from dopamine.metrics import statistics_instance
import flax
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tensorflow as tf


try:
  logging.warning((
      'Setting tf to CPU only, to avoid OOM. '
      'See https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html '
      'for more information.'
  ))
  tf.config.set_visible_devices([], 'GPU')
except tf.errors.NotFoundError:
  logging.info((
      'Unable to modify visible devices. '
      "If you don't have a GPU, this is expected."
  ))


gin.constant('sac_agent.IMAGE_DTYPE', onp.uint8)
gin.constant('sac_agent.STATE_DTYPE', onp.float32)


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def train(
    network_def: nn.Module,
    optim: optax.GradientTransformation,
    alpha_optim: optax.GradientTransformation,
    optimizer_state: jnp.ndarray,
    alpha_optimizer_state: jnp.ndarray,
    network_params: flax.core.FrozenDict,
    target_params: flax.core.FrozenDict,
    log_alpha: jnp.ndarray,
    key: jnp.ndarray,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    next_states: jnp.ndarray,
    rewards: jnp.ndarray,
    terminals: jnp.ndarray,
    cumulative_gamma: float,
    target_entropy: float,
    reward_scale_factor: float,
) -> Mapping[str, Any]:
  """Run the training step.

  Returns a list of updated values and losses.

  Args:
    network_def: The SAC network definition.
    optim: The SAC optimizer (which also wraps the SAC parameters).
    alpha_optim: The optimizer for alpha.
    optimizer_state: The SAC optimizer state.
    alpha_optimizer_state: The alpha optimizer state.
    network_params: Parameters for SAC's online network.
    target_params: The parameters for SAC's target network.
    log_alpha: Parameters for alpha network.
    key: An rng key to use for random action selection.
    states: A batch of states.
    actions: A batch of actions.
    next_states: A batch of next states.
    rewards: A batch of rewards.
    terminals: A batch of terminals.
    cumulative_gamma: The discount factor to use.
    target_entropy: The target entropy for the agent.
    reward_scale_factor: A factor by which to scale rewards.

  Returns:
    A mapping from string keys to values, including updated optimizers and
      training statistics.
  """
  # Get the models from all the optimizers.
  frozen_params = network_params  # For use in loss_fn without apply gradients

  batch_size = states.shape[0]
  actions = jnp.reshape(actions, (batch_size, -1))  # Flatten

  def loss_fn(
      params: flax.core.FrozenDict,
      log_alpha: flax.core.FrozenDict,
      state: jnp.ndarray,
      action: jnp.ndarray,
      reward: jnp.ndarray,
      next_state: jnp.ndarray,
      terminal: jnp.ndarray,
      rng: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Calculates the loss for one transition.

    Args:
      params: Parameters for the SAC network.
      log_alpha: SAC's log_alpha parameter.
      state: A single state vector.
      action: A single action vector.
      reward: A reward scalar.
      next_state: A next state vector.
      terminal: A terminal scalar.
      rng: An RNG key to use for sampling actions.

    Returns:
      A tuple containing 1) the combined SAC loss and 2) a mapping containing
        statistics from the loss step.
    """
    rng1, rng2 = jax.random.split(rng, 2)

    # J_Q(\theta) from equation (5) in paper.
    q_value_1, q_value_2 = network_def.apply(
        params, state, action, method=network_def.critic
    )
    q_value_1 = jnp.squeeze(q_value_1)
    q_value_2 = jnp.squeeze(q_value_2)

    target_outputs = network_def.apply(target_params, next_state, rng1, True)
    target_q_value_1, target_q_value_2 = target_outputs.critic
    target_q_value = jnp.squeeze(
        jnp.minimum(target_q_value_1, target_q_value_2)
    )

    alpha_value = jnp.exp(log_alpha)  # pytype: disable=wrong-arg-types  # numpy-scalars
    log_prob = target_outputs.actor.log_probability
    target = reward_scale_factor * reward + cumulative_gamma * (
        target_q_value - alpha_value * log_prob
    ) * (1.0 - terminal)
    target = jax.lax.stop_gradient(target)
    critic_loss_1 = losses.mse_loss(q_value_1, target)
    critic_loss_2 = losses.mse_loss(q_value_2, target)
    critic_loss = jnp.mean(critic_loss_1 + critic_loss_2)

    # J_{\pi}(\phi) from equation (9) in paper.
    mean_action, sampled_action, action_log_prob = network_def.apply(
        params, state, rng2, method=network_def.actor
    )

    # We use frozen_params so that gradients can flow back to the actor without
    # being used to update the critic.
    q_value_no_grad_1, q_value_no_grad_2 = network_def.apply(
        frozen_params, state, sampled_action, method=network_def.critic
    )
    no_grad_q_value = jnp.squeeze(
        jnp.minimum(q_value_no_grad_1, q_value_no_grad_2)
    )
    alpha_value = jnp.exp(jax.lax.stop_gradient(log_alpha))  # pytype: disable=wrong-arg-types  # numpy-scalars
    policy_loss = jnp.mean(alpha_value * action_log_prob - no_grad_q_value)

    # J(\alpha) from equation (18) in paper.
    entropy_diff = -action_log_prob - target_entropy
    alpha_loss = jnp.mean(log_alpha * jax.lax.stop_gradient(entropy_diff))

    # Giving a smaller weight to the critic empirically gives better results
    combined_loss = 0.5 * critic_loss + 1.0 * policy_loss + 1.0 * alpha_loss
    return combined_loss, {
        'critic_loss': critic_loss,
        'policy_loss': policy_loss,
        'alpha_loss': alpha_loss,
        'critic_value_1': q_value_1,
        'critic_value_2': q_value_2,
        'target_value_1': target_q_value_1,
        'target_value_2': target_q_value_2,
        'mean_action': mean_action,
    }

  grad_fn = jax.vmap(
      jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True),
      in_axes=(None, None, 0, 0, 0, 0, 0, 0),
  )

  rng = jnp.stack(jax.random.split(key, num=batch_size))
  (_, aux_vars), gradients = grad_fn(
      network_params,
      log_alpha,
      states,
      actions,
      rewards,
      next_states,
      terminals,
      rng,
  )

  # This calculates the mean gradient/aux_vars using the individual
  # gradients/aux_vars from each item in the batch.
  gradients = jax.tree.map(functools.partial(jnp.mean, axis=0), gradients)
  aux_vars = jax.tree.map(functools.partial(jnp.mean, axis=0), aux_vars)
  network_gradient, alpha_gradient = gradients

  # Apply gradients to all the optimizers.
  updates, optimizer_state = optim.update(
      network_gradient, optimizer_state, params=network_params
  )
  network_params = optax.apply_updates(network_params, updates)
  alpha_updates, alpha_optimizer_state = alpha_optim.update(
      alpha_gradient, alpha_optimizer_state, params=log_alpha
  )
  log_alpha = optax.apply_updates(log_alpha, alpha_updates)

  # Compile everything in a dict.
  returns = {
      'network_params': network_params,
      'log_alpha': log_alpha,
      'optimizer_state': optimizer_state,
      'alpha_optimizer_state': alpha_optimizer_state,
      'Losses/Critic': aux_vars['critic_loss'],
      'Losses/Actor': aux_vars['policy_loss'],
      'Losses/Alpha': aux_vars['alpha_loss'],
      'Values/CriticValues1': jnp.mean(aux_vars['critic_value_1']),
      'Values/CriticValues2': jnp.mean(aux_vars['critic_value_2']),
      'Values/TargetValues1': jnp.mean(aux_vars['target_value_1']),
      'Values/TargetValues2': jnp.mean(aux_vars['target_value_2']),
      'Values/Alpha': jnp.squeeze(jnp.exp(log_alpha)),
  }
  for i, a in enumerate(aux_vars['mean_action']):
    returns.update({f'Values/MeanActions{i}': a})
  return returns


@functools.partial(jax.jit, static_argnums=0)
def select_action(network_def, params, state, rng, eval_mode=False):
  """Sample an action to take from the current policy network.

  This obtains a mean and variance from the input policy network, and samples an
  action using a Gaussian distribution.

  Args:
    network_def: Linen Module to use for inference.
    params: Linen params (frozen dict) to use for inference.
    state: input state to use for inference.
    rng: Jax random number generator.
    eval_mode: bool, whether in eval mode.

  Returns:
    rng: Jax random number generator.
    action: int, the selected action.
  """
  rng, rng2 = jax.random.split(rng)
  greedy_a, sampled_a, _ = network_def.apply(
      params, state, rng2, method=network_def.actor
  )
  return rng, jnp.where(eval_mode, greedy_a, sampled_a)


@gin.configurable
class SACAgent(dqn_agent.JaxDQNAgent):
  """A JAX implementation of the SAC agent."""

  def __init__(
      self,
      action_shape,
      action_limits,
      observation_shape,
      action_dtype=jnp.float32,
      observation_dtype=jnp.float32,
      reward_scale_factor=1.0,
      stack_size=1,
      network=continuous_networks.ActorCriticNetwork,
      num_layers=2,
      hidden_units=256,
      gamma=0.99,
      update_horizon=1,
      min_replay_history=20000,
      update_period=1,
      target_update_type='soft',
      target_update_period=1000,
      target_smoothing_coefficient=0.005,
      target_entropy=None,
      eval_mode=False,
      optimizer='adam',
      summary_writer=None,
      summary_writing_frequency=500,
      allow_partial_reload=False,
      seed=None,
      collector_allowlist='tensorboard',
  ):
    r"""Initializes the agent and constructs the necessary components.

    Args:
      action_shape: int or tuple, dimensionality of the action space.
      action_limits: pair of lower and higher bounds for actions.
      observation_shape: tuple of ints describing the observation shape.
      action_dtype: jnp.dtype, specifies the type of the actions.
      observation_dtype: jnp.dtype, specifies the type of the observations.
      reward_scale_factor: float, factor by which to scale rewards.
      stack_size: int, number of frames to use in state stack.
      network: Jax network to use for training.
      num_layers: int, number of layers in the network.
      hidden_units: int, number of hidden units in the network.
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_type: str, if 'hard', will perform a hard update of the
        target network every target_update_period training steps; if 'soft',
        will use target_smoothing_coefficient to update the target network at
        every training step.
      target_update_period: int, frequency with which to update target network
        when in 'hard' mode.
      target_smoothing_coefficient: float, smoothing coefficient for target
        network updates (\tau in paper) when in 'soft' mode.
      target_entropy: float or None, the target entropy for training alpha. If
        None, it will default to the half the negative of the number of action
        dimensions.
      eval_mode: bool, True for evaluation and False for training.
      optimizer: str, name of optimizer to use.
      summary_writer: SummaryWriter object for outputting training statistics.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      allow_partial_reload: bool, whether we allow reloading a partial agent
        (for instance, only the network parameters).
      seed: int, a seed for SAC's internal RNG, used for initialization and
        sampling actions.
      collector_allowlist: list of str, if using CollectorDispatcher, this can
        be used to specify which Collectors to log to.
    """
    assert isinstance(observation_shape, tuple)
    # If we're performing hard updates, we force the smoothing coefficient to 1.
    if target_update_type == 'hard':
      target_smoothing_coefficient = 1.0

    if isinstance(action_shape, int):
      action_shape = (action_shape,)

    # If target_entropy is None, set to default value.
    if target_entropy is None:
      action_dim = functools.reduce(operator.mul, action_shape, 1.0)
      target_entropy = -0.5 * action_dim
    self._seed = int(time.time() * 1e6) if seed is None else seed
    logging.info(
        'Creating %s agent with the following parameters:',
        self.__class__.__name__,
    )
    logging.info('\t action_shape: %s', action_shape)
    logging.info('\t action_dtype: %s', action_dtype)
    logging.info('\t action_limits: %s', action_limits)
    logging.info('\t observation_shape: %s', observation_shape)
    logging.info('\t observation_dtype: %s', observation_dtype)
    logging.info('\t reward_scale_factor: %f', reward_scale_factor)
    logging.info('\t num_layers: %d', num_layers)
    logging.info('\t hidden_units: %d', hidden_units)
    logging.info('\t gamma: %f', gamma)
    logging.info('\t update_horizon: %f', update_horizon)
    logging.info('\t min_replay_history: %d', min_replay_history)
    logging.info('\t update_period: %d', update_period)
    logging.info('\t target_update_type: %s', target_update_type)
    logging.info('\t target_update_period: %d', target_update_period)
    logging.info(
        '\t target_smoothing_coefficient: %f', target_smoothing_coefficient
    )
    logging.info('\t target_entropy: %f', target_entropy)
    logging.info('\t optimizer: %s', optimizer)
    logging.info('\t seed: %d', self._seed)

    self.action_shape = action_shape
    self.action_dtype = action_dtype
    self.observation_shape = tuple(observation_shape)
    self.observation_dtype = observation_dtype
    self.reward_scale_factor = reward_scale_factor
    self.stack_size = stack_size
    self.action_limits = action_limits
    action_limits = tuple(tuple(x.reshape(-1)) for x in action_limits)
    self.network_def = network(
        action_shape, num_layers, hidden_units, action_limits
    )
    self.gamma = gamma
    self.update_horizon = update_horizon
    self.cumulative_gamma = math.pow(gamma, update_horizon)
    self.min_replay_history = min_replay_history
    self.update_period = update_period
    self.target_update_type = target_update_type
    self.target_update_period = target_update_period
    self.target_smoothing_coefficient = target_smoothing_coefficient
    self.target_entropy = target_entropy
    self.eval_mode = eval_mode
    self.training_steps = 0
    self.summary_writer = summary_writer
    self.summary_writing_frequency = summary_writing_frequency
    self.allow_partial_reload = allow_partial_reload
    self._collector_allowlist = collector_allowlist

    self._rng = jax.random.PRNGKey(self._seed)
    state_shape = self.observation_shape + (stack_size,)
    self.state = onp.zeros(state_shape)
    self._replay_scheme = 'uniform'
    self._replay = self._build_replay_buffer()
    self._optimizer_name = optimizer
    self._build_networks_and_optimizer()

    # Variables to be initialized by the agent once it interacts with the
    # environment.
    self._observation = None
    self._last_observation = None

  def _build_networks_and_optimizer(self):
    self._rng, init_key = jax.random.split(self._rng)

    # We can reuse init_key safely for the action selection key
    # since it is only used for shape inference during initialization.
    self.network_params = self.network_def.init(init_key, self.state, init_key)
    self.network_optimizer = dqn_agent.create_optimizer(self._optimizer_name)
    self.optimizer_state = self.network_optimizer.init(self.network_params)

    # TODO(joshgreaves): Find a way to just copy the critic params
    self.target_params = self.network_params

    # \alpha network
    self.log_alpha = jnp.zeros(1)
    self.alpha_optimizer = dqn_agent.create_optimizer(self._optimizer_name)
    self.alpha_optimizer_state = self.alpha_optimizer.init(self.log_alpha)

  def _build_replay_buffer(self):
    """Creates the replay buffer used by the agent."""
    transition_accumulator = accumulator.TransitionAccumulator(
        stack_size=self.stack_size,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
    )
    sampling_distribution = samplers.UniformSamplingDistribution(
        seed=self._seed
    )
    return replay_buffer.ReplayBuffer(
        transition_accumulator=transition_accumulator,
        sampling_distribution=sampling_distribution,
    )

  def _maybe_sync_weights(self):
    """Syncs the target weights with the online weights."""
    if (
        self.target_update_type == 'hard'
        and self.training_steps % self.target_update_period != 0
    ):
      return

    def _sync_weights(target_p, online_p):
      return (
          self.target_smoothing_coefficient * online_p
          + (1 - self.target_smoothing_coefficient) * target_p
      )

    self.target_params = jax.tree.map(
        _sync_weights, self.target_params, self.network_params
    )

  def begin_episode(self, observation):
    """Returns the agent's first action for this episode.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      np.ndarray, the selected action.
    """
    self._reset_state()
    self._record_observation(observation)

    if not self.eval_mode:
      self._train_step()

    if self._replay.add_count > self.min_replay_history:
      self._rng, self.action = select_action(
          self.network_def,
          self.network_params,
          self.state,
          self._rng,
          self.eval_mode,
      )
    else:
      self._rng, action_rng = jax.random.split(self._rng)
      self.action = jax.random.uniform(
          action_rng,
          self.action_shape,
          self.action_dtype,
          self.action_limits[0],
          self.action_limits[1],
      )
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

    if self._replay.add_count > self.min_replay_history:
      self._rng, self.action = select_action(
          self.network_def,
          self.network_params,
          self.state,
          self._rng,
          self.eval_mode,
      )
    else:
      self._rng, action_rng = jax.random.split(self._rng)
      self.action = jax.random.uniform(
          action_rng,
          self.action_shape,
          self.action_dtype,
          self.action_limits[0],
          self.action_limits[1],
      )
    self.action = onp.asarray(self.action)
    return self.action

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
        self._rng, key = jax.random.split(self._rng)

        train_returns = train(
            self.network_def,
            self.network_optimizer,
            self.alpha_optimizer,
            self.optimizer_state,
            self.alpha_optimizer_state,
            self.network_params,
            self.target_params,
            self.log_alpha,
            key,
            self.replay_elements['state'],
            self.replay_elements['action'],
            self.replay_elements['next_state'],
            self.replay_elements['reward'],
            self.replay_elements['terminal'],
            self.cumulative_gamma,
            self.target_entropy,
            self.reward_scale_factor,
        )

        self.network_params = train_returns['network_params']
        self.optimizer_state = train_returns['optimizer_state']
        self.log_alpha = train_returns['log_alpha']
        self.alpha_optimizer_state = train_returns['alpha_optimizer_state']

        if (
            self.summary_writer is not None
            and self.training_steps > 0
            and self.training_steps % self.summary_writing_frequency == 0
        ):

          statistics = []
          for k in train_returns:
            if k.startswith('Losses') or k.startswith('Values'):
              self.summary_writer.scalar(
                  k, train_returns[k], self.training_steps
              )
              if hasattr(self, 'collector_dispatcher'):
                statistics.append(
                    statistics_instance.StatisticsInstance(
                        k,
                        onp.asarray(train_returns[k]),
                        step=self.training_steps,
                    )
                )
          if hasattr(self, 'collector_dispatcher'):
            self.collector_dispatcher.write(
                statistics, collector_allowlist=self._collector_allowlist
            )
          self.summary_writer.flush()
        self._maybe_sync_weights()
    self.training_steps += 1

  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    """Returns a self-contained bundle of the agent's state.

    This is used for checkpointing. It will return a dictionary containing all
    non-TensorFlow objects (to be saved into a file by the caller), and it saves
    all TensorFlow objects into a checkpoint file.

    Args:
      checkpoint_dir: str, directory where TensorFlow objects will be saved.
      iteration_number: int, iteration number to use for naming the checkpoint
        file.

    Returns:
      A dict containing additional Python objects to be checkpointed by the
        experiment. If the checkpoint directory does not exist, returns None.
    """
    if not tf.io.gfile.exists(checkpoint_dir):
      return None
    # Checkpoint the out-of-graph replay buffer.
    self._replay.save(checkpoint_dir, iteration_number)
    bundle_dictionary = {
        'state': self.state,
        'training_steps': self.training_steps,
        'network_params': self.network_params,
        'optimizer_state': self.optimizer_state,
        'target_params': self.target_params,
        'log_alpha': self.log_alpha,
        'alpha_optimizer_state': self.alpha_optimizer_state,
    }
    return bundle_dictionary

  def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
    """Restores the agent from a checkpoint.

    Restores the agent's Python objects to those specified in bundle_dictionary,
    and restores the TensorFlow objects to those specified in the
    checkpoint_dir. If the checkpoint_dir does not exist, will not reset the
      agent's state.

    Args:
      checkpoint_dir: str, path to the checkpoint saved.
      iteration_number: int, checkpoint version, used when restoring the replay
        buffer.
      bundle_dictionary: dict, containing additional Python objects owned by the
        agent.

    Returns:
      bool, True if unbundling was successful.
    """
    try:
      # self._replay.load() will throw a NotFoundError if it does not find all
      # the necessary files.
      self._replay.load(checkpoint_dir, iteration_number)
    except tf.errors.NotFoundError:
      if not self.allow_partial_reload:
        # If we don't allow partial reloads, we will return False.
        return False
      logging.warning('Unable to reload replay buffer!')
    if bundle_dictionary is not None:
      self.state = bundle_dictionary['state']
      self.training_steps = bundle_dictionary['training_steps']

      self.network_params = bundle_dictionary['network_params']
      self.network_optimizer = dqn_agent.create_optimizer(self._optimizer_name)
      self.optimizer_state = bundle_dictionary['optimizer_state']
      self.target_params = bundle_dictionary['target_params']
      self.log_alpha = bundle_dictionary['log_alpha']
      self.alpha_optimizer_state = bundle_dictionary['alpha_optimizer_state']
    elif not self.allow_partial_reload:
      return False
    else:
      logging.warning("Unable to reload the agent's parameters!")
    return True

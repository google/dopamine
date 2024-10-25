# coding=utf-8
# Copyright 2024 The Dopamine Authors.
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
"""Compact implementation of a PPO agent in JAX.

Based on agent described in
  "Proximal Policy Optimization Algorithms"
  by John Schulman et al.
  https://arxiv.org/abs/1707.06347
"""

import functools
import time

from absl import logging
from dopamine.jax import continuous_networks
from dopamine.jax import losses
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.replay_memory import accumulator
from dopamine.jax.replay_memory import replay_buffer
from dopamine.jax.replay_memory import samplers
from dopamine.metrics import statistics_instance
import flax
from flax import linen as nn
from flax.metrics import tensorboard
import gin
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf


def train(
    network_def: nn.Module,
    network_params: flax.core.FrozenDict,
    optim: optax.GradientTransformation,
    optimizer_state: jnp.ndarray,
    key: jnp.ndarray,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    next_states: jnp.ndarray,
    rewards: jnp.ndarray,
    terminals: jnp.ndarray,
    num_epochs: int,
    batch_size: int,
    epsilon: float,
    gamma: float,
    lambda_: float,
    vf_coefficient: float,
    entropy_coefficient: float,
    clip_critic_loss: bool,
):
  """Run the training step.

  Args:
    network_def: The PPO network definition.
    network_params: Parameters for PPO's online network.
    optim: The PPO optimizer.
    optimizer_state: The PPO optimizer state.
    key: An rng key to use for random action selection.
    states: A trajectory of states.
    actions: A trajectory of actions.
    next_states: A trajectory of next states.
    rewards: A trajectory of rewards.
    terminals: A trajectory of terminals.
    num_epochs: The number of epochs to run.
    batch_size: The batch size to use.
    epsilon: The epsilon clipping for PPO.
    gamma: The discount factor.
    lambda_: The GAE parameter for calculating advantage function.
    vf_coefficient: The value function coefficient for PPO.
    entropy_coefficient: The entropy coefficient for PPO.
    clip_critic_loss: Whether to clip the critic loss.

  Returns:
    network_params: The updated network parameters.
    optimizer_state: The updated optimizer state.
    loss_stats: Training statistics.
  """
  q_values = jax.vmap(
      lambda state: network_def.apply(
          network_params, state, method=network_def.critic
      ).q_value
  )(states)
  q_values = jax.lax.stop_gradient(q_values)
  q_values = jnp.squeeze(q_values)

  # Next q value for the last collected state used in the delta (12) calcuation
  # to estimate future rewards beyond the current trajectory.
  # Need to apply whole network as next action has not be sampled yet.
  rng, key = jax.random.split(key)
  next_q_value = network_def.apply(
      network_params, next_states[-1], rng
  ).critic.q_value
  next_q_value = jax.lax.stop_gradient(next_q_value)
  next_q_value = jnp.squeeze(next_q_value)

  advantages, returns = calculate_advantages_and_returns(
      q_values,
      next_q_value,
      rewards,
      terminals,
      gamma,
      lambda_,
  )

  log_probability = jax.vmap(
      lambda state, action: network_def.apply(
          network_params, state, action=action, method=network_def.actor
      ).log_probability
  )(states, actions)
  log_probability = jax.lax.stop_gradient(log_probability)

  batch_keys = jnp.stack(jax.random.split(key, num=states.shape[0]))
  sampled_actions = jax.vmap(
      lambda state, key: network_def.apply(
          network_params, state, key=key, method=network_def.actor
      ).sampled_action
  )(states, batch_keys)
  sampled_actions = jnp.mean(sampled_actions, axis=0)

  (
      num_batches,
      states,
      actions,
      returns,
      advantages,
      log_probability,
      q_values,
  ) = create_minibatches_and_shuffle(
      states,
      actions,
      returns,
      advantages,
      log_probability,
      q_values,
      batch_size,
      key,
  )

  loss_stats = {
      'combined_loss': [],
      'actor_loss': [],
      'critic_loss': [],
      'entropy_loss': [],
  }
  for _ in range(num_epochs):
    for i in range(num_batches):
      network_params, optimizer_state, aux_vars = train_minibatch(
          network_def,
          network_params,
          optim,
          optimizer_state,
          states[i],
          actions[i],
          returns[i],
          advantages[i],
          log_probability[i],
          q_values[i],
          epsilon,
          vf_coefficient,
          entropy_coefficient,
          clip_critic_loss,
      )
      for k, v in aux_vars.items():
        if k in loss_stats:
          loss_stats[k].append(v.item())

  loss_stats = {
      'Losses/Combined': np.mean(loss_stats.get('combined_loss', 0.0)),
      'Losses/Actor': np.mean(loss_stats.get('actor_loss', 0.0)),
      'Losses/Critic': np.mean(loss_stats.get('critic_loss', 0.0)),
      'Losses/Entropy': np.mean(loss_stats.get('entropy_loss', 0.0)),
  }
  for i, a in enumerate(sampled_actions):
    loss_stats.update({f'Values/SampledAction{i}': a})
  return network_params, optimizer_state, loss_stats


def calculate_advantages_and_returns(
    q_values: jnp.ndarray,
    next_q_value: jnp.ndarray,
    rewards: jnp.ndarray,
    terminals: jnp.ndarray,
    gamma: float,
    lambda_: float,
):
  """Calculates the advantages and returns."""
  nsteps = rewards.shape[0]

  # Generalized Advantage Estimation calculated based on equations (11) and (12)
  advantages = np.zeros_like(rewards)
  for t in reversed(range(nsteps)):
    delta = (
        rewards[t]
        + gamma
        * (q_values[t + 1] if t < nsteps - 1 else next_q_value)
        * (1.0 - terminals[t])
        - q_values[t]
    )
    advantages[t] = delta + lambda_ * gamma * (
        advantages[t + 1] if t < nsteps - 1 else 0.0
    ) * (1.0 - terminals[t])
  # returns: Total discounted future rewards (R + yV(s'))
  # -> Calculated by adding V(s) to advantages (R + yV(s') - V(s))
  returns = advantages + q_values
  return jnp.array(advantages), jnp.array(returns)


def create_minibatches_and_shuffle(
    states: jnp.ndarray,
    actions: jnp.ndarray,
    returns: jnp.ndarray,
    advantages: jnp.ndarray,
    log_probability: jnp.ndarray,
    q_values: jnp.ndarray,
    batch_size: int,
    key: jnp.ndarray,
):
  """Create and shuffle minibatches for training."""
  assert (
      states.shape[0] % batch_size == 0
  ), 'Batch size must divide number of states.'
  num_batches = states.shape[0] // batch_size

  states = states.reshape((num_batches, batch_size) + states.shape[1:])
  actions = actions.reshape((num_batches, batch_size) + actions.shape[1:])
  returns = returns.reshape((num_batches, batch_size) + returns.shape[1:])
  advantages = advantages.reshape(
      (num_batches, batch_size) + advantages.shape[1:]
  )
  log_probability = log_probability.reshape(
      (num_batches, batch_size) + log_probability.shape[1:]
  )
  q_values = q_values.reshape((num_batches, batch_size) + q_values.shape[1:])

  indices = jax.random.permutation(key, num_batches)
  states = states[indices]
  actions = actions[indices]
  returns = returns[indices]
  advantages = advantages[indices]
  log_probability = log_probability[indices]
  q_values = q_values[indices]

  return (
      num_batches,
      states,
      actions,
      returns,
      advantages,
      log_probability,
      q_values,
  )


@functools.partial(
    jax.jit,
    static_argnames=[
        'network_def',
        'optim',
        'epsilon',
        'vf_coefficient',
        'entropy_coefficient',
        'clip_critic_loss',
    ],
)
def train_minibatch(
    network_def: nn.Module,
    network_params: flax.core.FrozenDict,
    optim: optax.GradientTransformation,
    optimizer_state: jnp.ndarray,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    returns: jnp.ndarray,
    advantages: jnp.ndarray,
    log_probability: jnp.ndarray,
    q_values: jnp.ndarray,
    epsilon: float,
    vf_coefficient: float,
    entropy_coefficient: float,
    clip_critic_loss: bool,
):
  """Run the training step for a minibatch."""

  def loss_fn(
      network_params: flax.core.FrozenDict,
      states: jnp.ndarray,
      actions: jnp.ndarray,
      returns: jnp.ndarray,
      advantages: jnp.ndarray,
      old_log_probability: jnp.ndarray,
      old_q_values: jnp.ndarray,
  ):
    """Calculates the loss of a minibatch."""
    actor_output = jax.vmap(
        lambda state, action: network_def.apply(
            network_params,
            state,
            action=action,
            method=network_def.actor,
        )
    )(states, actions)
    log_probability = actor_output.log_probability
    ratio = jnp.exp(log_probability - old_log_probability)
    actor_loss = jnp.mean(
        -jnp.minimum(
            ratio * advantages,
            jnp.clip(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages,
        )
    )

    q_values = jax.vmap(
        lambda state: network_def.apply(
            network_params, state, method=network_def.critic
        ).q_value
    )(states)
    q_values = jnp.squeeze(q_values)
    # Scale critic loss by 0.5 to match baselines implementation.
    if clip_critic_loss:
      # Value function loss clipping as specified in:
      # https://ppo-details.cleanrl.dev/2021/11/05/ppo-implementation-details/
      # Note: May hurt performance according to above
      critic_loss = 0.5 * jnp.mean(
          jnp.maximum(
              losses.mse_loss(q_values, returns),
              losses.mse_loss(
                  jnp.clip(
                      q_values, old_q_values - epsilon, old_q_values + epsilon
                  ),
                  returns,
              ),
          )
      )
    else:
      critic_loss = 0.5 * jnp.mean(losses.mse_loss(q_values, returns))

    entropy = actor_output.entropy if actor_output.entropy is not None else 0.0
    entropy_loss = jnp.mean(entropy)

    loss = (
        actor_loss
        + vf_coefficient * critic_loss
        - entropy_coefficient * entropy_loss
    )

    return loss, {
        'combined_loss': loss,
        'actor_loss': actor_loss,
        'critic_loss': critic_loss,
        'entropy_loss': entropy_loss,
    }

  grad_fn = jax.value_and_grad(loss_fn, argnums=(0), has_aux=True)

  # Normalize advantages within minibatch.
  advantages = jnp.array(
      (advantages - advantages.mean()) / (advantages.std() + 1e-8)
  )

  (_, aux_vars), grad = grad_fn(
      network_params,
      states,
      actions,
      returns,
      advantages,
      log_probability,
      q_values,
  )

  updates, optimizer_state = optim.update(
      grad, optimizer_state, params=network_params
  )
  network_params = optax.apply_updates(network_params, updates)

  return network_params, optimizer_state, aux_vars


@functools.partial(jax.jit, static_argnames=['network_def'])
def select_action(network_def, params, state, rng):
  """Sample an action to take from the current policy network.

  This obtains a mean and variance from the input policy network, and samples an
  action using a Gaussian distribution.

  Args:
    network_def: Linen Module to use for inference.
    params: Linen params (frozen dict) to use for inference.
    state: input state to use for inference.
    rng: Jax random number generator.

  Returns:
    rng: Jax random number generator.
    action: The selected action.
  """
  rng, rng2 = jax.random.split(rng)
  action = network_def.apply(
      params, state, rng2, method=network_def.actor
  ).sampled_action
  action = jax.lax.stop_gradient(action)
  return rng, action


@gin.configurable
class PPOAgent(dqn_agent.JaxDQNAgent):
  """A JAX implementation of the PPO agent."""

  def __init__(
      self,
      action_shape,
      observation_shape,
      action_limits,
      stack_size=1,
      update_horizon=1,
      network=continuous_networks.PPOActorCriticNetwork,
      num_layers=2,
      hidden_units=64,
      activation='tanh',
      update_period=2048,
      num_epochs=10,
      batch_size=64,
      gamma=0.99,
      lambda_=0.95,
      epsilon=0.2,
      vf_coefficient=0.5,
      entropy_coefficient=0.0,
      clip_critic_loss=True,
      eval_mode=False,
      optimizer='adam',
      max_gradient_norm=0.5,
      summary_writer=None,
      summary_writing_frequency=1,
      allow_partial_reload=False,
      seed=None,
      collector_allowlist='tensorboard',
  ):
    r"""Initializes the agent and constructs the necessary components.

    Args:
      action_shape: int or tuple, dimensionality of the action space.
      observation_shape: tuple of ints describing the observation shape.
      action_limits: pair of lower and higher bounds for actions.
      stack_size: int, number of frames to use in state stack.
      update_horizon: int, replay buffer update horizon.
      network: PPO network to use for training.
      num_layers: int, number of layers in the network.
      hidden_units: int, number of hidden units in the network.
      activation: nn.Module, activation function to use in the network.
      update_period: int, period between PPO updates.
      num_epochs: int, number of epochs to run the agent for.
      batch_size: int, number of elements to sample from the replay buffer.
      gamma: float, discount factor with the usual RL meaning.
      lambda_: float, GAE parameter for calculating advantage function.
      epsilon: float, epsilon clipping for PPO.
      vf_coefficient: float, value function coefficient for PPO.
      entropy_coefficient: float, entropy coefficient for PPO.
      clip_critic_loss: bool, whether to clip critic loss.
      eval_mode: bool, True for evaluation and False for training.
      optimizer: str, name of optimizer to use.
      max_gradient_norm: float, maximum gradient norm for optimizer.
      summary_writer: SummaryWriter object for outputting training statistics.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      allow_partial_reload: bool, whether we allow reloading a partial agent
        (for instance, only the network parameters).
      seed: int, a seed for PPO's internal RNG, used for initialization and
        sampling actions.
      collector_allowlist: list of str, if using CollectorDispatcher, this can
        be used to specify which Collectors to log to.
    """
    assert isinstance(observation_shape, tuple)
    self._seed = int(time.time() * 1e6) if seed is None else seed

    self._use_new_replay_buffer = True
    self._replay_scheme = 'uniform'

    if isinstance(action_shape, int):
      action_shape = (action_shape,)

    logging.info(
        'Creating %s agent with the following parameters:',
        self.__class__.__name__,
    )
    logging.info('\t action_shape: %s', action_shape)
    logging.info('\t action_limits: %s', action_limits)
    logging.info('\t observation_shape: %s', observation_shape)
    logging.info('\t num_layers: %d', num_layers)
    logging.info('\t hidden_units: %d', hidden_units)
    logging.info('\t update_period: %d', update_period)
    logging.info('\t num_epochs: %d', num_epochs)
    logging.info('\t batch_size: %d', batch_size)
    logging.info('\t gamma: %f', gamma)
    logging.info('\t lambda_: %f', lambda_)
    logging.info('\t epsilon: %f', epsilon)
    logging.info('\t vf_coefficient: %f', vf_coefficient)
    logging.info('\t entropy_coefficient: %f', entropy_coefficient)
    logging.info('\t clip_critic_loss: %s', clip_critic_loss)
    logging.info('\t optimizer: %s', optimizer)
    logging.info('\t seed: %d', self._seed)

    self.action_shape = action_shape
    if action_limits is not None:
      action_limits = tuple(tuple(x.reshape(-1)) for x in action_limits)
    self.observation_shape = tuple(observation_shape)
    if network.__name__ == 'PPOActorCriticNetwork':
      self.network_def = network(
          action_shape=action_shape,
          action_limits=action_limits,
          num_layers=num_layers,
          hidden_units=hidden_units,
          activation=continuous_networks.create_activation(activation),
      )
    else:
      self.network_def = network(
          action_shape=action_shape,
          action_limits=action_limits,
      )
    self.stack_size = stack_size
    self.update_horizon = update_horizon
    self.update_period = update_period
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.gamma = gamma
    self.lambda_ = lambda_
    self.epsilon = epsilon
    self.vf_coefficient = vf_coefficient
    self.entropy_coefficient = entropy_coefficient
    self.clip_critic_loss = clip_critic_loss
    self.eval_mode = eval_mode

    self.training_steps = 0
    if isinstance(summary_writer, str):
      try:
        tf.compat.v1.enable_v2_behavior()
      except ValueError:
        pass
      self.summary_writer = tf.summary.create_file_writer(summary_writer)
    else:
      self.summary_writer = summary_writer
    self.summary_writing_frequency = summary_writing_frequency
    self.allow_partial_reload = allow_partial_reload
    self._collector_allowlist = collector_allowlist

    self._rng = jax.random.PRNGKey(self._seed)
    state_shape = self.observation_shape + (self.stack_size,)
    self.state = np.zeros(state_shape)
    self._replay = self._build_replay_buffer()
    self._optimizer_name = optimizer
    self._max_gradient_norm = max_gradient_norm
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
    self.network_optimizer = optax.chain(
        # Perform global gradient clipping as specified in:
        # https://ppo-details.cleanrl.dev/2021/11/05/ppo-implementation-details
        optax.clip_by_global_norm(self._max_gradient_norm),
        dqn_agent.create_optimizer(self._optimizer_name),
    )
    self.optimizer_state = self.network_optimizer.init(self.network_params)

  def _build_replay_buffer(self):
    transition_accumulator = accumulator.TransitionAccumulator(
        stack_size=self.stack_size,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
    )
    sampling_distribution = samplers.SequentialSamplingDistribution(
        seed=self._seed,
        sort_samples=False,
    )
    return replay_buffer.ReplayBuffer(
        transition_accumulator=transition_accumulator,
        sampling_distribution=sampling_distribution,
    )

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

    self._rng, self.action = select_action(
        self.network_def,
        self.network_params,
        self.state,
        self._rng,
    )
    self.action = np.asarray(self.action)
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

    self._rng, self.action = select_action(
        self.network_def,
        self.network_params,
        self.state,
        self._rng,
    )
    self.action = np.asarray(self.action)
    return self.action

  def _train_step(self):
    """Runs a single training step."""
    if self._replay.add_count >= self.update_period:
      self._sample_from_replay_buffer()
      self._rng, key = jax.random.split(self._rng)

      self.network_params, self.optimizer_state, loss_stats = train(
          self.network_def,
          self.network_params,
          self.network_optimizer,
          self.optimizer_state,
          key,
          self.replay_elements['state'],
          self.replay_elements['action'],
          self.replay_elements['next_state'],
          self.replay_elements['reward'],
          self.replay_elements['terminal'],
          self.num_epochs,
          self.batch_size,
          self.epsilon,
          self.gamma,
          self.lambda_,
          self.vf_coefficient,
          self.entropy_coefficient,
          self.clip_critic_loss,
      )
      self._replay.clear()
      # re-add last next state for continuity
      self._record_observation(self._last_observation)

      if (
          self.summary_writer is not None
          and self.training_steps > 0
          and self.training_steps % self.summary_writing_frequency == 0
      ):
        statistics = []
        for k in loss_stats:
          if isinstance(self.summary_writer, tensorboard.SummaryWriter):
            self.summary_writer.scalar(k, loss_stats[k], self.training_steps)
          else:
            with self.summary_writer.as_default():
              tf.summary.scalar(k, loss_stats[k], self.training_steps)
          if hasattr(self, 'collector_dispatcher'):
            statistics.append(
                statistics_instance.StatisticsInstance(
                    k,
                    np.asarray(loss_stats[k]),
                    step=self.training_steps,
                )
            )
        if hasattr(self, 'collector_dispatcher'):
          self.collector_dispatcher.write(
              statistics, collector_allowlist=self._collector_allowlist
          )
        self.summary_writer.flush()
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
    elif not self.allow_partial_reload:
      return False
    else:
      logging.warning("Unable to reload the agent's parameters!")
    return True

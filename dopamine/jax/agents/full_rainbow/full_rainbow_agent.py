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
"""Compact implementation of the full Rainbow agent in JAX.

Specifically, we implement the following components from Rainbow:

  * n-step updates
  * prioritized replay
  * distributional RL
  * double_dqn
  * noisy
  * dueling

Details in "Rainbow: Combining Improvements in Deep Reinforcement Learning" by
Hessel et al. (2018).
"""

import functools

from absl import logging
from dopamine.jax import losses
from dopamine.jax import networks
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.agents.rainbow import rainbow_agent
from dopamine.jax.replay_memory import accumulator
from dopamine.jax.replay_memory import elements
from dopamine.jax.replay_memory import replay_buffer
from dopamine.jax.replay_memory import samplers
from dopamine.metrics import statistics_instance
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tensorflow as tf


@gin.configurable
def zero_epsilon(
    unused_decay_period, unused_step, unused_warmup_steps, unused_epsilon
):
  return 0.0


@functools.partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8, 10, 11))
def select_action(
    network_def,
    params,
    state,
    rng,
    num_actions,
    eval_mode,
    epsilon_eval,
    epsilon_train,
    epsilon_decay_period,
    training_steps,
    min_replay_history,
    epsilon_fn,
    support,
):
  """Select an action from the set of available actions."""
  epsilon = jnp.where(
      eval_mode,
      epsilon_eval,
      epsilon_fn(
          epsilon_decay_period,
          training_steps,
          min_replay_history,
          epsilon_train,
      ),
  )

  rng, rng1, rng2, rng3 = jax.random.split(rng, num=4)
  p = jax.random.uniform(rng1)
  best_actions = jnp.argmax(
      network_def.apply(
          params, state, key=rng2, eval_mode=eval_mode, support=support
      ).q_values
  )
  return rng, jnp.where(
      p <= epsilon, jax.random.randint(rng3, (), 0, num_actions), best_actions
  )


@functools.partial(jax.vmap, in_axes=(None, 0, None))
def get_logits(model, states, rng):
  return model(states, key=rng).logits


@functools.partial(jax.vmap, in_axes=(None, 0, None))
def get_q_values(model, states, rng):
  return model(states, key=rng).q_values


@functools.partial(
    jax.jit,
    static_argnames=(
        'network_def',
        'optimizer',
        'cumulative_gamma',
        'double_dqn',
        'distributional',
        'mse_loss',
    ),
)
def train(
    network_def,
    online_params,
    target_params,
    optimizer,
    optimizer_state,
    states,
    actions,
    next_states,
    rewards,
    terminals,
    loss_weights,
    support,
    cumulative_gamma,
    double_dqn,
    distributional,
    mse_loss,
    rng,
):
  """Run a training step."""

  # Split the current rng into 2 for updating the rng after this call
  rng, rng1, rng2 = jax.random.split(rng, num=3)

  def q_online(state, key):
    return network_def.apply(online_params, state, key=key, support=support)

  def q_target(state, key):
    return network_def.apply(target_params, state, key=key, support=support)

  def loss_fn(params, target, loss_multipliers):
    """Computes the distributional loss for C51 or huber loss for DQN."""

    def q_online(state, key):
      return network_def.apply(params, state, key=key, support=support)

    if distributional:
      logits = get_logits(q_online, states, rng)
      logits = jnp.squeeze(logits)
      # Fetch the logits for its selected action. We use vmap to perform this
      # indexing across the batch.
      chosen_action_logits = jax.vmap(lambda x, y: x[y])(logits, actions)
      loss = jax.vmap(losses.softmax_cross_entropy_loss_with_logits)(
          target, chosen_action_logits
      )
    else:
      q_values = get_q_values(q_online, states, rng)
      q_values = jnp.squeeze(q_values)
      replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)

      loss = losses.mse_loss if mse_loss else losses.huber_loss
      loss = jax.vmap(loss)(target, replay_chosen_q)

    mean_loss = jnp.mean(loss_multipliers * loss)
    return mean_loss, loss

  # Use the weighted mean loss for gradient computation.
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  target = target_output(
      q_online,
      q_target,
      next_states,
      rewards,
      terminals,
      support,
      cumulative_gamma,
      double_dqn,
      distributional,
      rng1,
  )

  # Get the unweighted loss without taking its mean for updating priorities.
  (mean_loss, loss), grad = grad_fn(online_params, target, loss_weights)
  updates, optimizer_state = optimizer.update(
      grad, optimizer_state, params=online_params
  )
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, loss, mean_loss, rng2


@functools.partial(
    jax.vmap, in_axes=(None, None, 0, 0, 0, None, None, None, None, None)
)
def target_output(
    model,
    target_network,
    next_states,
    rewards,
    terminals,
    support,
    cumulative_gamma,
    double_dqn,
    distributional,
    rng,
):
  """Builds the C51 target distribution or DQN target Q-values."""

  is_terminal_multiplier = 1.0 - terminals.astype(jnp.float32)
  # Incorporate terminal state to discount factor.
  gamma_with_terminal = cumulative_gamma * is_terminal_multiplier

  target_network_dist = target_network(next_states, key=rng)
  if double_dqn:
    # Use the current network for the action selection
    next_state_target_outputs = model(next_states, key=rng)
  else:
    next_state_target_outputs = target_network_dist
  # Action selection using Q-values for next-state
  q_values = jnp.squeeze(next_state_target_outputs.q_values)
  next_qt_argmax = jnp.argmax(q_values)

  if distributional:
    # Compute the target Q-value distribution
    probabilities = jnp.squeeze(target_network_dist.probabilities)
    next_probabilities = probabilities[next_qt_argmax]
    target_support = rewards + gamma_with_terminal * support
    target = rainbow_agent.project_distribution(
        target_support, next_probabilities, support
    )
  else:
    # Compute the target Q-value
    next_q_values = jnp.squeeze(target_network_dist.q_values)
    replay_next_qt_max = next_q_values[next_qt_argmax]
    target = rewards + gamma_with_terminal * replay_next_qt_max

  return jax.lax.stop_gradient(target)


@gin.configurable
class JaxFullRainbowAgent(dqn_agent.JaxDQNAgent):
  """A compact implementation of the full Rainbow agent."""

  def __init__(
      self,
      num_actions,
      noisy=True,
      dueling=True,
      double_dqn=True,
      distributional=True,
      mse_loss=False,
      num_updates_per_train_step=1,
      network=networks.FullRainbowNetwork,
      num_atoms=51,
      vmax=10.0,
      vmin=None,
      epsilon_fn=dqn_agent.linearly_decaying_epsilon,
      replay_scheme='prioritized',
      summary_writer=None,
      seed=None,
      preprocess_fn=None,
  ):
    """Initializes the agent and constructs the necessary components.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      noisy: bool, Whether to use noisy networks or not.
      dueling: bool, Whether to use dueling network architecture or not.
      double_dqn: bool, Whether to use Double DQN or not.
      distributional: bool, whether to use distributional RL or not.
      mse_loss: bool, mse loss function.
      num_updates_per_train_step: int, Number of gradient updates every training
        step. Defaults to 1.
      network: flax.linen Module, neural network used by the agent initialized
        by shape in _create_network below. See
        dopamine.jax.networks.RainbowNetwork as an example.
      num_atoms: int, the number of buckets of the value function distribution.
      vmax: float, the value distribution support is [vmin, vmax].
      vmin: float, the value distribution support is [vmin, vmax]. If vmin is
        None, it is set to -vmax.
      epsilon_fn: function expecting 4 parameters: (decay_period, step,
        warmup_steps, epsilon). This function should return the epsilon value
        used for exploration during training.
      replay_scheme: str, 'prioritized' or 'uniform', the sampling scheme of the
        replay memory.
      summary_writer: SummaryWriter object, for outputting training statistics.
      seed: int, a seed for Jax RNG and initialization.
      preprocess_fn: function expecting the input state as parameter which it
        preprocesses (such as normalizing the pixel values between 0 and 1)
        before passing it to the Q-network. Defaults to None.
    """
    logging.info(
        'Creating %s agent with the following parameters:',
        self.__class__.__name__,
    )
    logging.info('\t double_dqn: %s', double_dqn)
    logging.info('\t noisy_networks: %s', noisy)
    logging.info('\t dueling_dqn: %s', dueling)
    logging.info('\t distributional: %s', distributional)
    logging.info('\t mse_loss: %d', mse_loss)
    logging.info('\t num_atoms: %d', num_atoms)
    logging.info('\t replay_scheme: %s', replay_scheme)
    logging.info(
        '\t num_updates_per_train_step: %d', num_updates_per_train_step
    )
    # We need this because some tools convert round floats into ints.
    vmax = float(vmax)
    self._num_atoms = num_atoms
    vmin = vmin if vmin else -vmax
    self._support = jnp.linspace(vmin, vmax, num_atoms)
    self._replay_scheme = replay_scheme
    self._double_dqn = double_dqn
    self._noisy = noisy
    self._dueling = dueling
    self._distributional = distributional
    self._mse_loss = mse_loss
    self._num_updates_per_train_step = num_updates_per_train_step

    super().__init__(
        num_actions=num_actions,
        network=functools.partial(
            network,
            num_atoms=num_atoms,
            noisy=self._noisy,
            dueling=self._dueling,
            distributional=self._distributional,
        ),
        epsilon_fn=zero_epsilon if self._noisy else epsilon_fn,
        summary_writer=summary_writer,
        seed=seed,
        preprocess_fn=preprocess_fn,
    )

  def _build_networks_and_optimizer(self):
    self._rng, rng = jax.random.split(self._rng)
    state = self.preprocess_fn(self.state)
    self.online_params = self.network_def.init(
        rng, x=state, support=self._support
    )
    self.optimizer = dqn_agent.create_optimizer(self._optimizer_name)
    self.optimizer_state = self.optimizer.init(self.online_params)
    self.target_network_params = self.online_params

  def _build_replay_buffer(self):
    """Creates the replay buffer used by the agent."""
    if self._replay_scheme not in ['uniform', 'prioritized']:
      raise ValueError('Invalid replay scheme: {}'.format(self._replay_scheme))

    transition_accumulator = accumulator.TransitionAccumulator(
        stack_size=self.stack_size,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
    )
    sampling_distribution = samplers.PrioritizedSamplingDistribution(
        seed=self._seed
    )
    return replay_buffer.ReplayBuffer(
        transition_accumulator=transition_accumulator,
        sampling_distribution=sampling_distribution,
    )

  def _training_step_update(self):
    """Gradient update during every training step."""

    self._sample_from_replay_buffer()
    states = self.preprocess_fn(self.replay_elements['state'])
    next_states = self.preprocess_fn(self.replay_elements['next_state'])

    if self._replay_scheme == 'prioritized':
      # The original prioritized experience replay uses a linear exponent
      # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of
      # 0.5 on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders)
      # suggested a fixed exponent actually performs better, except on Pong.
      probs = self.replay_elements['sampling_probabilities']
      # Weight the loss by the inverse priorities.
      loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
      loss_weights /= jnp.max(loss_weights)
    else:
      # Uniform weights if not using prioritized replay.
      loss_weights = jnp.ones(states.shape[0])

    (self.optimizer_state, self.online_params, loss, mean_loss, self._rng) = (
        train(
            self.network_def,
            self.online_params,
            self.target_network_params,
            self.optimizer,
            self.optimizer_state,
            states,
            self.replay_elements['action'],
            next_states,
            self.replay_elements['reward'],
            self.replay_elements['terminal'],
            loss_weights,
            self._support,
            self.cumulative_gamma,
            self._double_dqn,
            self._distributional,
            self._mse_loss,
            self._rng,
        )
    )

    if self._replay_scheme == 'prioritized':
      # Rainbow and prioritized replay are parametrized by an exponent
      # alpha, but in both cases it is set to 0.5 - for simplicity's sake we
      # leave it as is here, using the more direct sqrt(). Taking the square
      # root "makes sense", as we are dealing with a squared loss.  Add a
      # small nonzero value to the loss to avoid 0 priority items. While
      # technically this may be okay, setting all items to 0 priority will
      # cause troubles, and also result in 1.0 / 0.0 = NaN correction terms.
      self._replay.update(
          self.replay_elements['indices'],
          priorities=jnp.sqrt(loss + 1e-10),
      )

    if (
        self.summary_writer is not None
        and self.training_steps > 0
        and self.training_steps % self.summary_writing_frequency == 0
    ):
      with self.summary_writer.as_default():
        tf.summary.scalar(
            'CrossEntropyLoss', mean_loss, step=self.training_steps
        )
      self.summary_writer.flush()
      if hasattr(self, 'collector_dispatcher'):
        self.collector_dispatcher.write(
            [
                statistics_instance.StatisticsInstance(
                    'Loss', onp.asarray(mean_loss), step=self.training_steps
                ),
            ],
            collector_allowlist=self._collector_allowlist,
        )

  def _store_transition(
      self,
      last_observation,
      action,
      reward,
      is_terminal,
      *args,
      priority=None,
      episode_end=False
  ):
    """Stores a transition when in training mode."""
    # pylint: disable=protected-access
    is_prioritized = isinstance(
        self._replay._sampling_distribution,
        samplers.PrioritizedSamplingDistribution,
    )
    if is_prioritized and priority is None:
      if self._replay_scheme == 'uniform':
        priority = 1.0
      else:
        priority = (
            self._replay._sampling_distribution._sum_tree.max_recorded_priority
        )
    # pylint: enable=protected-access

    if not self.eval_mode:
      self._replay.add(
          elements.TransitionElement(
              last_observation,
              action,
              reward,
              is_terminal,
              episode_end,
          ),
          priority=priority,
          *args,
      )

  def _train_step(self):
    """Runs a single training step.

    Runs training if both:
      (1) A minimum number of frames have been added to the replay buffer.
      (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online_network_params to target_network_params if
    training steps is a multiple of target update period.
    """
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        for _ in range(self._num_updates_per_train_step):
          self._training_step_update()

      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1

  def begin_episode(self, observation):
    """Returns the agent's first action for this episode."""
    self._reset_state()
    self._record_observation(observation)

    if not self.eval_mode:
      self._train_step()

    state = self.preprocess_fn(self.state)
    self._rng, self.action = select_action(
        self.network_def,
        self.online_params,
        state,
        self._rng,
        self.num_actions,
        self.eval_mode,
        self.epsilon_eval,
        self.epsilon_train,
        self.epsilon_decay_period,
        self.training_steps,
        self.min_replay_history,
        self.epsilon_fn,
        self._support,
    )
    self.action = onp.asarray(self.action)
    return self.action

  def step(self, reward, observation):
    """Records the most recent transition and returns the agent's next action."""
    self._last_observation = self._observation
    self._record_observation(observation)

    if not self.eval_mode:
      self._store_transition(self._last_observation, self.action, reward, False)
      self._train_step()

    state = self.preprocess_fn(self.state)
    self._rng, self.action = select_action(
        self.network_def,
        self.online_params,
        state,
        self._rng,
        self.num_actions,
        self.eval_mode,
        self.epsilon_eval,
        self.epsilon_train,
        self.epsilon_decay_period,
        self.training_steps,
        self.min_replay_history,
        self.epsilon_fn,
        self._support,
    )
    self.action = onp.asarray(self.action)
    return self.action

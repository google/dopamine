# coding=utf-8
# Copyright 2021 The Atari 100k Precipice Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An implementation of SPR in Jax.

Includes the features included in the full Rainbow agent.  Designed to work with
an optimized replay buffer that returns subsequences rather than individual
transitions.

Some details differ from the original implementation due to differences in
the underlying Rainbow implementations.  In particular:
* Dueling networks in Dopamine separate at the final layer, not the penultimate
  layer as in the original.
* Dopamine's prioritized experience replay does not decay its exponent over time.

We find that these changes do not drastically impact the overall performance of
the algorithm, however.

Details on Rainbow are available in
"Rainbow: Combining Improvements in Deep Reinforcement Learning" by Hessel et
al. (2018).  For details on SPR, see
"Data-Efficient Reinforcement Learning with Self-Predictive Representations" by
Schwarzer et al (2021).
"""

import collections
import copy
import functools
import time

from absl import logging
from dopamine.jax import losses
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.agents.rainbow import rainbow_agent as dopamine_rainbow_agent
from dopamine.replay_memory import prioritized_replay_buffer
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import tensorflow as tf
from dopamine.labs.atari_100k import spr_networks as networks
from dopamine.labs.atari_100k.replay_memory import time_batch_replay_buffer as tdrbs


@functools.partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8, 10, 11))
def select_action(network_def, params, state, rng, num_actions, eval_mode,
                  epsilon_eval, epsilon_train, epsilon_decay_period,
                  training_steps, min_replay_history, epsilon_fn, support):
  """Select an action from the set of available actions."""
  epsilon = jnp.where(
      eval_mode, epsilon_eval,
      epsilon_fn(epsilon_decay_period, training_steps, min_replay_history,
                 epsilon_train))

  rng, rng1, rng2, rng3 = jax.random.split(rng, num=4)
  p = jax.random.uniform(rng1)
  q_values = network_def.apply(
      params, state, key=rng2, eval_mode=eval_mode, support=support).q_values

  best_actions = jnp.argmax(q_values, axis=-1)
  return rng, jnp.where(p <= epsilon,
                        jax.random.randint(rng3, (), 0, num_actions),
                        best_actions)


@functools.partial(
    jax.vmap, in_axes=(None, 0, 0, None, None), axis_name="batch")
def get_logits(model, states, actions, do_rollout, rng):
  results = model(states, actions=actions, do_rollout=do_rollout, key=rng)[0]
  return results.logits, results.latent


@functools.partial(
    jax.vmap, in_axes=(None, 0, 0, None, None), axis_name="batch")
def get_q_values(model, states, actions, do_rollout, rng):
  results = model(states, actions=actions, do_rollout=do_rollout, key=rng)[0]
  return results.q_values, results.latent


@functools.partial(jax.vmap, in_axes=(None, 0, None), axis_name="batch")
def get_spr_targets(model, states, key):
  results = model(states, key)
  return results


@functools.partial(jax.jit, static_argnums=(0, 11, 12, 13, 15))
def train(network_def, target_params, optimizer, states, actions, next_states,
          rewards, terminals, same_traj_mask, loss_weights, support,
          cumulative_gamma, double_dqn, distributional, rng, spr_weight):
  """Run a training step."""

  current_state = states[:, 0]
  online_params = optimizer.target
  # Split the current rng into 2 for updating the rng after this call
  rng, rng1, rng2 = jax.random.split(rng, num=3)
  use_spr = spr_weight > 0

  def q_online(state, key, actions=None, do_rollout=False):
    return network_def.apply(
        online_params,
        state,
        actions=actions,
        do_rollout=do_rollout,
        key=key,
        support=support,
        mutable=["batch_stats"])

  def q_target(state, key):
    return network_def.apply(
        target_params, state, key=key, support=support, mutable=["batch_stats"])

  def encode_project(state, key):
    latent, _ = network_def.apply(
        target_params,
        state,
        method=network_def.encode,
        mutable=["batch_stats"])
    latent = latent.reshape(-1)
    return network_def.apply(
        target_params,
        latent,
        key=key,
        eval_mode=True,
        method=network_def.project)

  def loss_fn(params, target, spr_targets, loss_multipliers):
    """Computes the distributional loss for C51 or huber loss for DQN."""

    def q_online(state, key, actions=None, do_rollout=False):
      return network_def.apply(
          params,
          state,
          actions=actions,
          do_rollout=do_rollout,
          key=key,
          support=support,
          mutable=["batch_stats"])

    if distributional:
      (logits, spr_predictions) = get_logits(q_online, current_state,
                                             actions[:, :-1], use_spr, rng)
      logits = jnp.squeeze(logits)
      # Fetch the logits for its selected action. We use vmap to perform this
      # indexing across the batch.
      chosen_action_logits = jax.vmap(lambda x, y: x[y])(logits, actions[:, 0])
      dqn_loss = jax.vmap(losses.softmax_cross_entropy_loss_with_logits)(
          target, chosen_action_logits)
    else:
      q_values, spr_predictions = get_q_values(q_online, current_state,
                                               actions[:, :-1], use_spr, rng)
      q_values = jnp.squeeze(q_values)
      replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions[:, 0])
      dqn_loss = jax.vmap(losses.huber_loss)(target, replay_chosen_q)

    if use_spr:
      spr_predictions = spr_predictions.transpose(1, 0, 2)
      spr_predictions = spr_predictions / jnp.linalg.norm(
          spr_predictions, 2, -1, keepdims=True)
      spr_targets = spr_targets / jnp.linalg.norm(
          spr_targets, 2, -1, keepdims=True)
      spr_loss = jnp.power(spr_predictions - spr_targets, 2).sum(-1)
      spr_loss = (spr_loss * same_traj_mask.transpose(1, 0)).mean(0)
    else:
      spr_loss = 0

    loss = dqn_loss + spr_weight * spr_loss

    mean_loss = jnp.mean(loss_multipliers * loss)
    return mean_loss, (loss, dqn_loss, spr_loss)

  # Use the weighted mean loss for gradient computation.
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  target = target_output(q_online, q_target, next_states, rewards, terminals,
                         support, cumulative_gamma, double_dqn, distributional,
                         rng1)

  if use_spr:
    future_states = states[:, 1:]
    spr_targets = get_spr_targets(
        encode_project, future_states.reshape(-1, *future_states.shape[2:]),
        rng1)
    spr_targets = spr_targets.reshape(*future_states.shape[:2],
                                      *spr_targets.shape[1:]).transpose(
                                          1, 0, 2)
  else:
    spr_targets = None

  # Get the unweighted loss without taking its mean for updating priorities.
  (mean_loss, (loss, dqn_loss,
               spr_loss)), grad = grad_fn(optimizer.target, target, spr_targets,
                                          loss_weights)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss, mean_loss, dqn_loss, spr_loss, rng2


@functools.partial(
    jax.vmap,
    in_axes=(None, None, 0, 0, 0, None, None, None, None, None),
    axis_name="batch")
def target_output(model, target_network, next_states, rewards, terminals,
                  support, cumulative_gamma, double_dqn, distributional, rng):
  """Builds the C51 target distribution or DQN target Q-values."""
  is_terminal_multiplier = 1. - terminals.astype(jnp.float32)
  # Incorporate terminal state to discount factor.
  gamma_with_terminal = cumulative_gamma * is_terminal_multiplier

  target_network_dist, _ = target_network(next_states, key=rng)
  if double_dqn:
    # Use the current network for the action selection
    next_state_target_outputs, _ = model(next_states, key=rng)
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
    target = dopamine_rainbow_agent.project_distribution(
        target_support, next_probabilities, support)
  else:
    # Compute the target Q-value
    next_q_values = jnp.squeeze(target_network_dist.q_values)
    replay_next_qt_max = next_q_values[next_qt_argmax]
    target = rewards + gamma_with_terminal * replay_next_qt_max

  return jax.lax.stop_gradient(target)


@gin.configurable
class SPRAgent(dqn_agent.JaxDQNAgent):
  """A compact implementation of SPR in Jax."""

  def __init__(self,
               num_actions,
               noisy=False,
               dueling=False,
               double_dqn=False,
               distributional=True,
               data_augmentation=False,
               num_updates_per_train_step=2,
               network=networks.SPRNetwork,
               num_atoms=51,
               vmax=10.,
               vmin=None,
               jumps=5,
               spr_weight=5,
               log_every=1,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               replay_scheme='prioritized',
               replay_type='deterministic',
               summary_writer=None,
               seed=None):
    """Initializes the agent and constructs the necessary components.

        Args:
            num_actions: int, number of actions the agent can take at any state.
            noisy: bool, Whether to use noisy networks or not.
            dueling: bool, Whether to use dueling network architecture or not.
            double_dqn: bool, Whether to use Double DQN or not.
            distributional: bool, whether to use distributional RL or not.
            data_augmentation: bool, Whether to use data augmentation or not.
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
            replay_type: str, 'deterministic' or 'regular', specifies the type of
                replay buffer to create.
            summary_writer: SummaryWriter object, for outputting training statistics.
            seed: int, a seed for Jax RNG and initialization.
        """
    logging.info('Creating %s agent with the following parameters:',
                 self.__class__.__name__)
    logging.info('\t double_dqn: %s', double_dqn)
    logging.info('\t noisy_networks: %s', noisy)
    logging.info('\t dueling_dqn: %s', dueling)
    logging.info('\t distributional: %s', distributional)
    logging.info('\t data_augmentation: %s', data_augmentation)
    logging.info('\t replay_scheme: %s', replay_scheme)
    logging.info('\t num_updates_per_train_step: %d',
                 num_updates_per_train_step)
    # We need this because some tools convert round floats into ints.
    vmax = float(vmax)
    self._num_atoms = num_atoms
    vmin = vmin if vmin else -vmax
    self._support = jnp.linspace(vmin, vmax, num_atoms)
    self._replay_scheme = replay_scheme
    self._replay_type = replay_type
    self._double_dqn = double_dqn
    self._noisy = noisy
    self._dueling = dueling
    self._distributional = distributional
    self._data_augmentation = data_augmentation
    self._num_updates_per_train_step = num_updates_per_train_step
    self._jumps = jumps
    self.spr_weight = spr_weight
    self.log_every = log_every
    super().__init__(
        num_actions=num_actions,
        network=functools.partial(
            network,
            num_atoms=num_atoms,
            noisy=self._noisy,
            dueling=self._dueling,
            distributional=self._distributional),
        epsilon_fn=dqn_agent.identity_epsilon if self._noisy else epsilon_fn,
        summary_writer=summary_writer,
        seed=seed)

  def _build_networks_and_optimizer(self):
    self._rng, rng = jax.random.split(self._rng)
    online_network_params = self.network_def.init(
        rng,
        x=self.state,
        actions=jnp.zeros((5,)),
        do_rollout=self.spr_weight > 0,
        support=self._support)
    optimizer_def = dqn_agent.create_optimizer(self._optimizer_name)
    self.optimizer = optimizer_def.create(online_network_params)
    self.target_network_params = copy.deepcopy(online_network_params)

  def _build_replay_buffer(self):
    """Creates the replay buffer used by the agent."""
    if self._replay_scheme not in ['uniform', 'prioritized']:
      raise ValueError('Invalid replay scheme: {}'.format(self._replay_scheme))
    if self._replay_type not in ['deterministic']:
      raise ValueError('Invalid replay type: {}'.format(self._replay_type))
    if self._replay_scheme == "prioritized":
      return tdrbs.DeterministicOutOfGraphPrioritizedTemporalReplayBuffer(
          observation_shape=self.observation_shape,
          stack_size=self.stack_size,
          update_horizon=self.update_horizon,
          gamma=self.gamma,
          jumps=self._jumps + 1,
          observation_dtype=self.observation_dtype,
      )
    else:
      return tdrbs.DeterministicOutOfGraphTemporalReplayBuffer(
          observation_shape=self.observation_shape,
          stack_size=self.stack_size,
          update_horizon=self.update_horizon,
          gamma=self.gamma,
          jumps=self._jumps + 1,
          observation_dtype=self.observation_dtype,
      )

  def _sample_from_replay_buffer(self):
    self._rng, rng = jax.random.split(self._rng)
    samples = self._replay.sample_transition_batch(rng)
    types = self._replay.get_transition_elements()
    self.replay_elements = collections.OrderedDict()
    for element, element_type in zip(samples, types):
      self.replay_elements[element_type.name] = element

  def _training_step_update(self):
    """Gradient update during every training step."""
    self._sample_from_replay_buffer()

    # Add code for data augmentation.
    self._rng, rng1, rng2 = jax.random.split(self._rng, num=3)
    states = networks.process_inputs(
        self.replay_elements['state'],
        rng=rng1,
        data_augmentation=self._data_augmentation)
    next_states = networks.process_inputs(
        self.replay_elements['next_state'][:, 0],
        rng=rng2,
        data_augmentation=self._data_augmentation)

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

    self.optimizer, loss, mean_loss, dqn_loss, spr_loss, self._rng = train(
        self.network_def, self.target_network_params, self.optimizer, states,
        self.replay_elements['action'], next_states,
        self.replay_elements['reward'][:,
                                       0], self.replay_elements['terminal'][:,
                                                                            0],
        self.replay_elements['same_trajectory'][:, 1:], loss_weights,
        self._support, self.cumulative_gamma, self._double_dqn,
        self._distributional, self._rng, self.spr_weight)

    if self._replay_scheme == 'prioritized':
      # Rainbow and prioritized replay are parametrized by an exponent
      # alpha, but in both cases it is set to 0.5 - for simplicity's sake we
      # leave it as is here, using the more direct sqrt(). Taking the square
      # root "makes sense", as we are dealing with a squared loss.  Add a
      # small nonzero value to the loss to avoid 0 priority items. While
      # technically this may be okay, setting all items to 0 priority will
      # cause troubles, and also result in 1.0 / 0.0 = NaN correction terms.
      self._replay.set_priority(self.replay_elements['indices'],
                                jnp.sqrt(dqn_loss + 1e-10))

    if self.summary_writer is not None:
      summary = tf.compat.v1.Summary(value=[
          tf.compat.v1.Summary.Value(
              tag='TotalLoss', simple_value=float(mean_loss)),
          tf.compat.v1.Summary.Value(
              tag='DQNLoss', simple_value=float(dqn_loss.mean())),
          tf.compat.v1.Summary.Value(
              tag='SPRLoss', simple_value=float(spr_loss.mean()))
      ])
      self.summary_writer.add_summary(summary, self.training_steps)

  def _store_transition(self,
                        last_observation,
                        action,
                        reward,
                        is_terminal,
                        *args,
                        priority=None,
                        episode_end=False):
    """Stores a transition when in training mode."""
    is_prioritized = (
        isinstance(self._replay,
                   prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer)
        or isinstance(
            self._replay,
            tdrbs.DeterministicOutOfGraphPrioritizedTemporalReplayBuffer))
    if is_prioritized and priority is None:
      if self._replay_scheme == 'uniform':
        priority = 1.
      else:
        priority = self._replay.sum_tree.max_recorded_priority

    if not self.eval_mode:
      self._replay.add(
          last_observation,
          action,
          reward,
          is_terminal,
          *args,
          priority=priority,
          episode_end=episode_end)

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

    state = networks.process_inputs(self.state, data_augmentation=False)
    self._rng, self.action = select_action(
        self.network_def, self.online_params, state, self._rng,
        self.num_actions, self.eval_mode, self.epsilon_eval, self.epsilon_train,
        self.epsilon_decay_period, self.training_steps, self.min_replay_history,
        self.epsilon_fn, self._support)

    self.action = onp.asarray(self.action)
    return self.action

  def step(self, reward, observation):
    """Records the most recent transition and returns the agent's next action."""
    self._last_observation = self._observation
    self._record_observation(observation)

    if not self.eval_mode:
      self._store_transition(self._last_observation, self.action, reward, False)
      self._train_step()

    state = networks.process_inputs(self.state, data_augmentation=False)
    self._rng, self.action = select_action(
        self.network_def, self.online_params, state, self._rng,
        self.num_actions, self.eval_mode, self.epsilon_eval, self.epsilon_train,
        self.epsilon_decay_period, self.training_steps, self.min_replay_history,
        self.epsilon_fn, self._support)
    self.action = onp.asarray(self.action)
    return self.action

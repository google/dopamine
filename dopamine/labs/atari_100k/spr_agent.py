# coding=utf-8
# Copyright 2023 The Dopamine Authors.
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
"""An implementation of SPR in Jax.

Includes the features included in the full Rainbow agent.  Designed to work with
an optimized replay buffer that returns subsequences rather than individual
transitions.
Some details differ from the original implementation due to differences in
the underlying Rainbow implementations.  In particular:
* Dueling networks in Dopamine separate at the final layer, not the penultimate
  layer as in the original.
* Dopamine's prioritized experience replay does not decay its exponent
  over time.
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
from dopamine.labs.atari_100k import atari_100k_rainbow_agent
from dopamine.labs.atari_100k.replay_memory import subsequence_replay_buffer as replay_buffers
from dopamine.labs.atari_100k.spr_networks import SPRNetwork
import gin
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf


@functools.partial(
    jax.vmap, in_axes=(None, 0, 0, None, None), axis_name='batch'
)
def get_logits(model, states, actions, do_rollout, rng):
  results = model(states, actions=actions, do_rollout=do_rollout, key=rng)[0]
  return results.logits, results.latent


@functools.partial(
    jax.vmap, in_axes=(None, 0, 0, None, None), axis_name='batch'
)
def get_q_values(model, states, actions, do_rollout, rng):
  results = model(states, actions=actions, do_rollout=do_rollout, key=rng)[0]
  return results.q_values, results.latent


@functools.partial(jax.vmap, in_axes=(None, 0, None), axis_name='batch')
def get_spr_targets(model, states, key):
  results = model(states, key)
  return results


@functools.partial(
    jax.jit,
    static_argnames=(
        'network_def',
        'optimizer',
        'double_dqn',
        'distributional',
        'spr_weight',
        'cumulative_gamma',
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
    same_traj_mask,
    loss_weights,
    support,
    cumulative_gamma,
    double_dqn,
    distributional,
    rng,
    spr_weight,
):
  """Run a training step."""

  current_state = states[:, 0]
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
        mutable=['batch_stats'],
    )

  def q_target(state, key):
    return network_def.apply(
        target_params, state, key=key, support=support, mutable=['batch_stats']
    )

  def encode_project(state, key):
    latent, _ = network_def.apply(
        target_params, state, method=network_def.encode, mutable=['batch_stats']
    )
    latent = latent.reshape(-1)
    return network_def.apply(
        target_params,
        latent,
        key=key,
        eval_mode=True,
        method=network_def.project,
    )

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
          mutable=['batch_stats'],
      )

    if distributional:
      (logits, spr_predictions) = get_logits(
          q_online, current_state, actions[:, :-1], use_spr, rng
      )
      logits = jnp.squeeze(logits)
      # Fetch the logits for its selected action. We use vmap to perform this
      # indexing across the batch.
      chosen_action_logits = jax.vmap(lambda x, y: x[y])(logits, actions[:, 0])
      dqn_loss = jax.vmap(losses.softmax_cross_entropy_loss_with_logits)(
          target, chosen_action_logits
      )
    else:
      q_values, spr_predictions = get_q_values(
          q_online, current_state, actions[:, :-1], use_spr, rng
      )
      q_values = jnp.squeeze(q_values)
      replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions[:, 0])
      dqn_loss = jax.vmap(losses.huber_loss)(target, replay_chosen_q)

    if use_spr:
      # transpose to move from (time, batch, latent_dim) to
      # (batch, time, latent_dim) to match targets
      spr_predictions = spr_predictions.transpose(1, 0, 2)

      # Calculate SPR loss (normalized L2)
      spr_predictions = spr_predictions / jnp.linalg.norm(
          spr_predictions, 2, -1, keepdims=True
      )
      spr_targets = spr_targets / jnp.linalg.norm(
          spr_targets, 2, -1, keepdims=True
      )

      spr_loss = jnp.power(spr_predictions - spr_targets, 2).sum(-1)

      # Zero out loss for predictions that cross into the next episode
      spr_loss = (spr_loss * same_traj_mask.transpose(1, 0)).mean(0)
    else:
      spr_loss = 0

    loss = dqn_loss + spr_weight * spr_loss

    mean_loss = jnp.mean(loss_multipliers * loss)
    return mean_loss, (loss, dqn_loss, spr_loss)

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

  if use_spr:
    future_states = states[:, 1:]
    spr_targets = get_spr_targets(
        encode_project,
        future_states.reshape(-1, *future_states.shape[2:]),
        rng1,
    )
    spr_targets = spr_targets.reshape(
        *future_states.shape[:2], *spr_targets.shape[1:]
    ).transpose(1, 0, 2)
  else:
    spr_targets = None

  # Get the unweighted loss without taking its mean for updating priorities.
  (mean_loss, (_, dqn_loss, spr_loss)), grad = grad_fn(
      online_params, target, spr_targets, loss_weights
  )
  updates, optimizer_state = optimizer.update(grad, optimizer_state)
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, mean_loss, dqn_loss, spr_loss, rng2


@functools.partial(
    jax.vmap,
    in_axes=(None, None, 0, 0, 0, None, None, None, None, None),
    axis_name='batch',
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
        target_support, next_probabilities, support
    )
  else:
    # Compute the target Q-value
    next_q_values = jnp.squeeze(target_network_dist.q_values)
    replay_next_qt_max = next_q_values[next_qt_argmax]
    target = rewards + gamma_with_terminal * replay_next_qt_max

  return jax.lax.stop_gradient(target)


@gin.configurable
class SPRAgent(atari_100k_rainbow_agent.Atari100kRainbowAgent):
  """A compact implementation of SPR in Jax."""

  def __init__(
      self,
      num_actions,
      jumps=5,
      spr_weight=5,
      summary_writer=None,
      seed=None,
      epsilon_fn=dqn_agent.linearly_decaying_epsilon,
      network=SPRNetwork,
  ):
    """Initializes the agent and constructs the necessary components.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      jumps: int >= 0, number of SPR prediction steps to do.
      spr_weight: float, weight given to the SPR loss.
      summary_writer: SummaryWriter object, for outputting training statistics.
      seed: int, a seed for Jax RNG and initialization.
      epsilon_fn: Type of epsilon decay to use. By default, linearly_decaying
        will use e-greedy during initial data collection, matching the PyTorch
        codebase.
      network: Network class to use.
    """
    logging.info(
        'Creating %s agent with the following parameters:',
        self.__class__.__name__,
    )
    logging.info('\t spr_weight: %s', spr_weight)
    logging.info('\t jumps: %s', jumps)
    self._jumps = jumps
    self.spr_weight = spr_weight
    super().__init__(
        num_actions=num_actions,
        summary_writer=summary_writer,
        seed=seed,
        network=network,
    )

    # Parent class JaxFullRainbowAgent will overwrite this with the wrong value,
    # so just reverse its change.
    self.epsilon_fn = epsilon_fn
    self.start_time = time.time()

  def _build_networks_and_optimizer(self):
    self._rng, rng = jax.random.split(self._rng)
    self.online_params = self.network_def.init(
        rng,
        x=self.state,
        actions=jnp.zeros((5,)),
        do_rollout=self.spr_weight > 0,
        support=self._support,
    )
    self.optimizer = dqn_agent.create_optimizer(self._optimizer_name)
    self.optimizer_state = self.optimizer.init(self.online_params)
    self.target_network_params = copy.deepcopy(self.online_params)

  def _build_replay_buffer(self):
    """Creates the replay buffer used by the agent."""
    if self._replay_scheme not in ['uniform', 'prioritized']:
      raise ValueError('Invalid replay scheme: {}'.format(self._replay_scheme))
    if self._replay_scheme == 'prioritized':
      return replay_buffers.PrioritizedJaxSubsequenceParallelEnvReplayBuffer(
          observation_shape=self.observation_shape,
          stack_size=self.stack_size,
          update_horizon=self.update_horizon,
          gamma=self.gamma,
          subseq_len=self._jumps + 1,
          observation_dtype=self.observation_dtype,
      )
    else:
      return replay_buffers.JaxSubsequenceParallelEnvReplayBuffer(
          observation_shape=self.observation_shape,
          stack_size=self.stack_size,
          update_horizon=self.update_horizon,
          gamma=self.gamma,
          subseq_len=self._jumps + 1,
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

    inter_batch_time = time.time() - self.start_time
    self.start_time = time.time()

    sample_start_time = time.time()
    self._sample_from_replay_buffer()
    sample_time = time.time() - sample_start_time

    aug_start_time = time.time()
    # Add code for data augmentation.
    self._rng, rng1, rng2 = jax.random.split(self._rng, num=3)
    states = self.train_preprocess_fn(self.replay_elements['state'], rng=rng1)
    next_states = self.train_preprocess_fn(
        self.replay_elements['next_state'][:, 0], rng=rng2
    )

    aug_time = time.time() - aug_start_time
    train_start_time = time.time()

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

    (
        self.optimizer_state,
        self.online_params,
        mean_loss,
        dqn_loss,
        spr_loss,
        self._rng,
    ) = train(
        network_def=self.network_def,
        online_params=self.online_params,
        target_params=self.target_network_params,
        optimizer=self.optimizer,
        optimizer_state=self.optimizer_state,
        states=states,
        actions=self.replay_elements['action'],
        next_states=next_states,
        rewards=self.replay_elements['reward'][:, 0],
        terminals=self.replay_elements['terminal'][:, 0],
        same_traj_mask=self.replay_elements['same_trajectory'][:, 1:],
        loss_weights=loss_weights,
        support=self._support,
        cumulative_gamma=self.cumulative_gamma,
        double_dqn=self._double_dqn,
        distributional=self._distributional,
        rng=self._rng,
        spr_weight=self.spr_weight,
    )

    if self._replay_scheme == 'prioritized':
      # Rainbow and prioritized replay are parametrized by an exponent
      # alpha, but in both cases it is set to 0.5 - for simplicity's sake we
      # leave it as is here, using the more direct sqrt(). Taking the square
      # root "makes sense", as we are dealing with a squared loss.  Add a
      # small nonzero value to the loss to avoid 0 priority items. While
      # technically this may be okay, setting all items to 0 priority will
      # cause troubles, and also result in 1.0 / 0.0 = NaN correction terms.
      self._replay.set_priority(
          self.replay_elements['indices'], jnp.sqrt(dqn_loss + 1e-10)
      )

    train_time = time.time() - train_start_time
    if (
        self.summary_writer is not None
        and self.training_steps > 0
        and self.training_steps % self.summary_writing_frequency == 0
    ):
      step = self.training_steps
      tf.summary.scalar('TotalLoss', float(mean_loss), step=step)
      tf.summary.scalar('DQNLoss', float(dqn_loss.mean()), step=step)
      tf.summary.scalar('SPRLoss', float(spr_loss.mean()), step=step)
      tf.summary.scalar('InterbatchTime', float(inter_batch_time), step=step)
      tf.summary.scalar('TrainTime', float(train_time), step=step)
      tf.summary.scalar('SampleTime', float(sample_time), step=step)
      tf.summary.scalar('AugTime', float(aug_time), step=step)
      self.summary_writer.flush()

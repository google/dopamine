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
"""Compact implementation of a simplified Rainbow agent in Jax.

Specifically, we implement the following components from Rainbow:

  * n-step updates;
  * prioritized replay; and
  * distributional RL.

These three components were found to significantly impact the performance of
the Atari game-playing agent.

Furthermore, our implementation does away with some minor hyperparameter
choices. Specifically, we

  * keep the beta exponent fixed at beta=0.5, rather than increase it linearly;
  * remove the alpha parameter, which was set to alpha=0.5 throughout the paper.

Details in "Rainbow: Combining Improvements in Deep Reinforcement Learning" by
Hessel et al. (2018).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from dopamine.jax import networks
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.replay_memory import prioritized_replay_buffer
from flax import nn
import gin
import jax
import jax.numpy as jnp
import tensorflow as tf


@jax.jit
def train(online_network, target, replay_elements, optimizer):
  """Run a training step."""
  def loss_fn(model, mean_loss=True):
    # size of indices: batch_size x 1.
    logits = model(replay_elements['state']).logits
    indices = jnp.arange(logits.shape[0])
    # For each element of the batch, fetch the logits for its selected action.
    chosen_action_logits = logits[indices, replay_elements['action'], :]
    loss = networks.softmax_cross_entropy_loss_with_logits(
        target,
        chosen_action_logits)
    if mean_loss:
      loss = jnp.mean(loss)
    return loss
  grad_fn = jax.value_and_grad(loss_fn)
  mean_loss, grad = grad_fn(online_network)
  # Get the loss without taking its mean.
  loss = loss_fn(online_network, mean_loss=False)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss, mean_loss


@gin.configurable
class JaxRainbowAgent(dqn_agent.JaxDQNAgent):
  """A compact implementation of a simplified Rainbow agent."""

  def __init__(self,
               num_actions,
               observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
               stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
               network=networks.RainbowNetwork,
               num_atoms=51,
               vmax=10.,
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
               allow_partial_reload=False):
    """Initializes the agent and constructs the necessary components.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints or an int. If single int, the observation
        is assumed to be a 2D square.
      observation_dtype: DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to jnp.float32.
      stack_size: int, number of frames to use in state stack.
      network: flax.nn Module that is initialized by shape in _create_network
        below. See dopamine.jax.networks.RainbowNetwork as an example.
      num_atoms: int, the number of buckets of the value function distribution.
      vmax: float, the value distribution support is [-vmax, vmax].
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
      allow_partial_reload: bool, whether we allow reloading a partial agent
        (for instance, only the network parameters).
    """
    # We need this because some tools convert round floats into ints.
    vmax = float(vmax)
    self._num_atoms = num_atoms
    self._support = jnp.linspace(-vmax, vmax, num_atoms)
    self._replay_scheme = replay_scheme

    super(JaxRainbowAgent, self).__init__(
        num_actions=num_actions,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        stack_size=stack_size,
        network=network.partial(num_atoms=num_atoms,
                                support=self._support),
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
        allow_partial_reload=allow_partial_reload)

  def _create_network(self, name):
    """Builds a convolutional network that outputs Q-value distributions.

    Args:
      name: str, this name is passed to the Jax Module.
    Returns:
      network: Jax Model, the network instantiated by Jax.
    """
    _, initial_params = self.network.init_by_shape(
        self._rng, name=name,
        input_specs=[(self.state.shape, self.observation_dtype)],
        num_actions=self.num_actions,
        num_atoms=self._num_atoms,
        support=self._support)
    return nn.Model(self.network, initial_params)

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

  def _build_target_distribution(self):
    """Builds the C51 target distribution as per Bellemare et al. (2017).

    First, we compute the support of the Bellman target, r + gamma Z'. Where Z'
    is the support of the next state distribution:

      * Evenly spaced in [-vmax, vmax] if the current state is nonterminal;
      * 0 otherwise (duplicated num_atoms times).

    Second, we compute the next-state probabilities, corresponding to the action
    with highest expected value.

    Finally we project the Bellman target (support + probabilities) onto the
    original support.

    Returns:
      The target distribution from the replay.
    """
    batch_size = self.replay_elements['reward'].shape[0]

    # size of rewards: batch_size x 1
    rewards = self.replay_elements['reward'][:, None]

    # size of tiled_support: batch_size x num_atoms
    tiled_support = jnp.tile(self._support, [batch_size])
    tiled_support = jnp.reshape(tiled_support, [batch_size, self._num_atoms])

    # size of target_support: batch_size x num_atoms
    is_terminal_multiplier = (
        1. - self.replay_elements['terminal'].astype(jnp.float32))
    # Incorporate terminal state to discount factor.
    # size of gamma_with_terminal: batch_size x 1
    gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
    gamma_with_terminal = gamma_with_terminal[:, None]
    target_support = rewards + gamma_with_terminal * tiled_support

    # size of next_qt_argmax: batch_size x 1
    next_state_target_outputs = self.target_network(
        self.replay_elements['next_state'])
    next_qt_argmax = jnp.argmax(next_state_target_outputs.q_values,
                                axis=1)[:, None]
    batch_indices = jnp.arange(batch_size)[:, None]

    # size of next_probabilities: batch_size x num_atoms
    next_probabilities = next_state_target_outputs.probabilities[
        batch_indices, next_qt_argmax, :].squeeze()

    return project_distribution(target_support, next_probabilities,
                                self._support)

  def _train_step(self):
    """Runs a single training step.

    Runs a training op if both:
      (1) A minimum number of frames have been added to the replay buffer.
      (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online_network to target_network if training steps
    is a multiple of target update period.
    """
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()
        self.optimizer, loss, mean_loss = train(
            online_network=self.online_network,
            target=jax.lax.stop_gradient(self._build_target_distribution()),
            replay_elements=self.replay_elements,
            optimizer=self.optimizer)
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
        if self.summary_writer is not None:
          summary = tf.compat.v1.Summary(value=[
              tf.compat.v1.Summary.Value(tag='CrossEntropyLoss',
                                         simple_value=mean_loss)])
          self.summary_writer.add_summary(summary, self.training_steps)
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1

  def _store_transition(self,
                        last_observation,
                        action,
                        reward,
                        is_terminal,
                        priority=None):
    """Stores a transition when in training mode.

    Stores the following tuple in the replay buffer (last_observation, action,
    reward, is_terminal, priority).

    Args:
      last_observation: Last observation, type determined via observation_type
        parameter in the replay_memory constructor.
      action: An integer, the action taken.
      reward: A float, the reward.
      is_terminal: Boolean indicating if the current state is a terminal state.
      priority: Float. Priority of sampling the transition. If None, the default
        priority will be used. If replay scheme is uniform, the default priority
        is 1. If the replay scheme is prioritized, the default priority is the
        maximum ever seen [Schaul et al., 2015].
    """
    if priority is None:
      if self._replay_scheme == 'uniform':
        priority = 1.
      else:
        priority = self._replay.sum_tree.max_recorded_priority

    if not self.eval_mode:
      self._replay.add(last_observation, action, reward, is_terminal, priority)


def project_distribution(supports, weights, target_support):
  """Projects a batch of (support, weights) onto target_support.

  Based on equation (7) in (Bellemare et al., 2017):
    https://arxiv.org/abs/1707.06887
  In the rest of the comments we will refer to this equation simply as Eq7.

  This code is not easy to digest, so we will use a running example to clarify
  what is going on, with the following sample inputs:

    * supports =       [[0, 2, 4, 6, 8],
                        [1, 3, 4, 5, 6]]
    * weights =        [[0.1, 0.6, 0.1, 0.1, 0.1],
                        [0.1, 0.2, 0.5, 0.1, 0.1]]
    * target_support = [4, 5, 6, 7, 8]

  In the code below, comments preceded with 'Ex:' will be referencing the above
  values.

  Args:
    supports: Jax array of shape (batch_size, num_dims) defining supports for
      the distribution.
    weights: Jax array of shape (batch_size, num_dims) defining weights on the
      original support points. Although for the CategoricalDQN agent these
      weights are probabilities, it is not required that they are.
    target_support: Jax array of shape (num_dims) defining support of the
      projected distribution. The values must be monotonically increasing. Vmin
      and Vmax will be inferred from the first and last elements of this Jax
      array, respectively. The values in this Jax array must be equally spaced.

  Returns:
    A Jax array of shape (batch_size, num_dims) with the projection of a batch
    of (support, weights) onto target_support.

  Raises:
    ValueError: If target_support has no dimensions, or if shapes of supports,
      weights, and target_support are incompatible.
  """
  target_support_deltas = target_support[1:] - target_support[:-1]
  # delta_z = `\Delta z` in Eq7.
  delta_z = target_support_deltas[0]
  # Ex: `v_min, v_max = 4, 8`.
  v_min, v_max = target_support[0], target_support[-1]
  # Ex: `batch_size = 2`.
  batch_size = supports.shape[0]
  # `N` in Eq7.
  # Ex: `num_dims = 5`.
  num_dims = target_support.shape[0]
  # clipped_support = `[\hat{T}_{z_j}]^{V_max}_{V_min}` in Eq7.
  # Ex: `clipped_support = [[[ 4.  4.  4.  6.  8.]]
  #                         [[ 4.  4.  4.  5.  6.]]]`.
  clipped_support = jnp.clip(supports, v_min, v_max)[:, None, :]
  # Ex: `tiled_support = [[[[ 4.  4.  4.  6.  8.]
  #                         [ 4.  4.  4.  6.  8.]
  #                         [ 4.  4.  4.  6.  8.]
  #                         [ 4.  4.  4.  6.  8.]
  #                         [ 4.  4.  4.  6.  8.]]
  #                        [[ 4.  4.  4.  5.  6.]
  #                         [ 4.  4.  4.  5.  6.]
  #                         [ 4.  4.  4.  5.  6.]
  #                         [ 4.  4.  4.  5.  6.]
  #                         [ 4.  4.  4.  5.  6.]]]]`.
  tiled_support = jnp.tile([clipped_support], [1, 1, num_dims, 1])
  # Ex: `reshaped_target_support = [[[ 4.]
  #                                  [ 5.]
  #                                  [ 6.]
  #                                  [ 7.]
  #                                  [ 8.]]
  #                                 [[ 4.]
  #                                  [ 5.]
  #                                  [ 6.]
  #                                  [ 7.]
  #                                  [ 8.]]]`.
  reshaped_target_support = jnp.tile(target_support[:, None], [batch_size, 1])
  reshaped_target_support = jnp.reshape(reshaped_target_support,
                                        [batch_size, num_dims, 1])
  # numerator = `|clipped_support - z_i|` in Eq7.
  # Ex: `numerator = [[[[ 0.  0.  0.  2.  4.]
  #                     [ 1.  1.  1.  1.  3.]
  #                     [ 2.  2.  2.  0.  2.]
  #                     [ 3.  3.  3.  1.  1.]
  #                     [ 4.  4.  4.  2.  0.]]
  #                    [[ 0.  0.  0.  1.  2.]
  #                     [ 1.  1.  1.  0.  1.]
  #                     [ 2.  2.  2.  1.  0.]
  #                     [ 3.  3.  3.  2.  1.]
  #                     [ 4.  4.  4.  3.  2.]]]]`.
  numerator = jnp.abs(tiled_support - reshaped_target_support)
  quotient = 1 - (numerator / delta_z)
  # clipped_quotient = `[1 - numerator / (\Delta z)]_0^1` in Eq7.
  # Ex: `clipped_quotient = [[[[ 1.  1.  1.  0.  0.]
  #                            [ 0.  0.  0.  0.  0.]
  #                            [ 0.  0.  0.  1.  0.]
  #                            [ 0.  0.  0.  0.  0.]
  #                            [ 0.  0.  0.  0.  1.]]
  #                           [[ 1.  1.  1.  0.  0.]
  #                            [ 0.  0.  0.  1.  0.]
  #                            [ 0.  0.  0.  0.  1.]
  #                            [ 0.  0.  0.  0.  0.]
  #                            [ 0.  0.  0.  0.  0.]]]]`.
  clipped_quotient = jnp.clip(quotient, 0, 1)
  # Ex: `weights = [[ 0.1  0.6  0.1  0.1  0.1]
  #                 [ 0.1  0.2  0.5  0.1  0.1]]`.
  weights = weights[:, None, :]
  # inner_prod = `\sum_{j=0}^{N-1} clipped_quotient * p_j(x', \pi(x'))`
  # in Eq7.
  # Ex: `inner_prod = [[[[ 0.1  0.6  0.1  0.  0. ]
  #                      [ 0.   0.   0.   0.  0. ]
  #                      [ 0.   0.   0.   0.1 0. ]
  #                      [ 0.   0.   0.   0.  0. ]
  #                      [ 0.   0.   0.   0.  0.1]]
  #                     [[ 0.1  0.2  0.5  0.  0. ]
  #                      [ 0.   0.   0.   0.1 0. ]
  #                      [ 0.   0.   0.   0.  0.1]
  #                      [ 0.   0.   0.   0.  0. ]
  #                      [ 0.   0.   0.   0.  0. ]]]]`.
  inner_prod = clipped_quotient * weights
  # Ex: `projection = [[ 0.8 0.0 0.1 0.0 0.1]
  #                    [ 0.8 0.1 0.1 0.0 0.0]]`.
  projection = jnp.sum(inner_prod, 3)
  projection = jnp.reshape(projection, [batch_size, num_dims])
  return projection

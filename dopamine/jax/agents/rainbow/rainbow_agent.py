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

import functools

from dopamine.jax import losses
from dopamine.jax import networks
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.replay_memory import accumulator
from dopamine.jax.replay_memory import replay_buffer
from dopamine.jax.replay_memory import samplers
from dopamine.metrics import statistics_instance
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tensorflow as tf


@functools.partial(jax.jit, static_argnums=(0, 3, 12))
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
):
  """Run a training step."""

  def loss_fn(params, target, loss_multipliers):
    def q_online(state):
      return network_def.apply(params, state, support)

    logits = jax.vmap(q_online)(states).logits
    # Fetch the logits for its selected action. We use vmap to perform this
    # indexing across the batch.
    chosen_action_logits = jax.vmap(lambda x, y: x[y])(logits, actions)
    loss = jax.vmap(losses.softmax_cross_entropy_loss_with_logits)(
        target, chosen_action_logits
    )
    mean_loss = jnp.mean(loss_multipliers * loss)
    return mean_loss, loss

  def q_target(state):
    return network_def.apply(target_params, state, support)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  target = target_distribution(
      q_target, next_states, rewards, terminals, support, cumulative_gamma
  )

  # Get the unweighted loss without taking its mean for updating priorities.
  (mean_loss, loss), grad = grad_fn(online_params, target, loss_weights)
  updates, optimizer_state = optimizer.update(
      grad, optimizer_state, params=online_params
  )
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, loss, mean_loss


@functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, None, None))
def target_distribution(
    target_network, next_states, rewards, terminals, support, cumulative_gamma
):
  """Builds the C51 target distribution as per Bellemare et al. (2017).

  First, we compute the support of the Bellman target, r + gamma Z'. Where Z'
  is the support of the next state distribution:

    * Evenly spaced in [-vmax, vmax] if the current state is nonterminal;
    * 0 otherwise (duplicated num_atoms times).

  Second, we compute the next-state probabilities, corresponding to the action
  with highest expected value.

  Finally we project the Bellman target (support + probabilities) onto the
  original support.

  Args:
    target_network: Jax Module used for the target network.
    next_states: numpy array of batched next states.
    rewards: numpy array of batched rewards.
    terminals: numpy array of batched terminals.
    support: support for the distribution.
    cumulative_gamma: float, cumulative gamma to use.

  Returns:
    The target distribution from the replay.
  """
  is_terminal_multiplier = 1.0 - terminals.astype(jnp.float32)
  # Incorporate terminal state to discount factor.
  gamma_with_terminal = cumulative_gamma * is_terminal_multiplier
  target_support = rewards + gamma_with_terminal * support
  next_state_target_outputs = target_network(next_states)
  q_values = jnp.squeeze(next_state_target_outputs.q_values)
  next_qt_argmax = jnp.argmax(q_values)
  probabilities = jnp.squeeze(next_state_target_outputs.probabilities)
  next_probabilities = probabilities[next_qt_argmax]
  return jax.lax.stop_gradient(
      project_distribution(target_support, next_probabilities, support)
  )


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
  """Select an action from the set of available actions.

  Chooses an action randomly with probability self._calculate_epsilon(), and
  otherwise acts greedily according to the current Q-value estimates.

  Args:
    network_def: Linen Module to use for inference.
    params: Linen params (frozen dict) to use for inference.
    state: input state to use for inference.
    rng: Jax random number generator.
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
    support: support for the distribution.

  Returns:
    rng: Jax random number generator.
    action: int, the selected action.
  """
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

  rng, rng1, rng2 = jax.random.split(rng, num=3)
  p = jax.random.uniform(rng1)
  return rng, jnp.where(
      p <= epsilon,
      jax.random.randint(rng2, (), 0, num_actions),
      jnp.argmax(network_def.apply(params, state, support).q_values),
  )


@gin.configurable
class JaxRainbowAgent(dqn_agent.JaxDQNAgent):
  """A compact implementation of a simplified Rainbow agent."""

  def __init__(
      self,
      num_actions,
      observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
      observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
      stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
      network=networks.RainbowNetwork,
      num_atoms=51,
      vmin=None,
      vmax=10.0,
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
      seed=None,
      summary_writer=None,
      summary_writing_frequency=500,
      allow_partial_reload=False,
  ):
    """Initializes the agent and constructs the necessary components.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints or an int. If single int, the observation
        is assumed to be a 2D square.
      observation_dtype: DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to jnp.float32.
      stack_size: int, number of frames to use in state stack.
      network: flax.linen Module that is initialized by shape in _create_network
        below. See dopamine.jax.networks.RainbowNetwork as an example.
      num_atoms: int, the number of buckets of the value function distribution.
      vmin: float, the value distribution support is [vmin, vmax]. If None, we
        set it to be -vmax.
      vmax: float, the value distribution support is [vmin, vmax].
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters: (decay_period, step,
        warmup_steps, epsilon). This function should return the epsilon value
        used for exploration during training.
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      replay_scheme: str, 'prioritized' or 'uniform', the sampling scheme of the
        replay memory.
      optimizer: str, name of optimizer to use.
      seed: int, a seed for Rainbow's internal RNG, used for initialization and
        sampling actions. If None, will use the current time in nanoseconds.
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
    # If vmin is not specified, set it to -vmax similar to C51.
    vmin = vmin if vmin else -vmax
    self._support = jnp.linspace(vmin, vmax, num_atoms)
    self._replay_scheme = replay_scheme

    super(JaxRainbowAgent, self).__init__(
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
        seed=seed,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency,
        allow_partial_reload=allow_partial_reload,
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

  # TODO(psc): Refactor this so we have a class _select_action that calls
  # select_action with the right parameters. This will allow us to avoid
  # overriding begin_episode.
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
        self.online_params,
        self.preprocess_fn(self.state),
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
    # TODO(psc): Why a numpy array? Why not an int?
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

    self._rng, self.action = select_action(
        self.network_def,
        self.online_params,
        self.preprocess_fn(self.state),
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
          loss_weights = jnp.ones(self.replay_elements['state'].shape[0])

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
            loss_weights,
            self._support,
            self.cumulative_gamma,
        )

        if self._replay_scheme == 'prioritized':
          # Rainbow and prioritized replay are parametrized by an exponent
          # alpha, but in both cases it is set to 0.5 - for simplicity's sake
          # we leave it as is here, using the more direct sqrt(). Taking the
          # square root "makes sense", as we are dealing with a squared loss.
          # Add a small nonzero value to the loss to avoid 0 priority items.
          # While technically this may be okay, setting all items to 0
          # priority will cause troubles, and also result in 1.0 / 0.0 = NaN
          # correction terms.
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
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1


def project_distribution(supports, weights, target_support):
  """Projects a batch of (support, weights) onto target_support.

  Based on equation (7) in (Bellemare et al., 2017):
    https://arxiv.org/abs/1707.06887
  In the rest of the comments we will refer to this equation simply as Eq7.

  Args:
    supports: Jax array of shape (num_dims) defining supports for the
      distribution.
    weights: Jax array of shape (num_dims) defining weights on the original
      support points. Although for the CategoricalDQN agent these weights are
      probabilities, it is not required that they are.
    target_support: Jax array of shape (num_dims) defining support of the
      projected distribution. The values must be monotonically increasing. Vmin
      and Vmax will be inferred from the first and last elements of this Jax
      array, respectively. The values in this Jax array must be equally spaced.

  Returns:
    A Jax array of shape (num_dims) with the projection of a batch
    of (support, weights) onto target_support.

  Raises:
    ValueError: If target_support has no dimensions, or if shapes of supports,
      weights, and target_support are incompatible.
  """
  v_min, v_max = target_support[0], target_support[-1]
  # `N` in Eq7.
  num_dims = target_support.shape[0]
  # delta_z = `\Delta z` in Eq7.
  delta_z = (v_max - v_min) / (num_dims - 1)
  # clipped_support = `[\hat{T}_{z_j}]^{V_max}_{V_min}` in Eq7.
  clipped_support = jnp.clip(supports, v_min, v_max)
  # numerator = `|clipped_support - z_i|` in Eq7.
  numerator = jnp.abs(clipped_support - target_support[:, None])
  quotient = 1 - (numerator / delta_z)
  # clipped_quotient = `[1 - numerator / (\Delta z)]_0^1` in Eq7.
  clipped_quotient = jnp.clip(quotient, 0, 1)
  # inner_prod = `\sum_{j=0}^{N-1} clipped_quotient * p_j(x', \pi(x'))` in Eq7.
  inner_prod = clipped_quotient * weights
  return jnp.squeeze(jnp.sum(inner_prod, -1))

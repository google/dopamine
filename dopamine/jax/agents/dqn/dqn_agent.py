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
"""Compact implementation of a DQN agent in JAx."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import inspect
import math
import time

from absl import logging

from dopamine.agents.dqn import dqn_agent
from dopamine.jax import networks
from dopamine.replay_memory import circular_replay_buffer
from dopamine.replay_memory import prioritized_replay_buffer
from flax import nn
from flax import optim
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import tensorflow as tf


NATURE_DQN_OBSERVATION_SHAPE = dqn_agent.NATURE_DQN_OBSERVATION_SHAPE
NATURE_DQN_DTYPE = jnp.uint8
NATURE_DQN_STACK_SIZE = dqn_agent.NATURE_DQN_STACK_SIZE
identity_epsilon = dqn_agent.identity_epsilon


@gin.configurable
def create_optimizer(name='adam', learning_rate=6.25e-5, beta1=0.9, beta2=0.999,
                     eps=1.5e-4, centered=False):
  """Create an optimizer for training.

  Currently, only the Adam optimizer is supported.

  Args:
    name: str, name of the optimizer to create.
    learning_rate: float, learning rate to use in the optimizer.
    beta1: float, beta1 parameter for the optimizer.
    beta2: float, beta2 parameter for the optimizer.
    eps: float, epsilon parameter for the optimizer.
    centered: bool, centered parameter for RMSProp.

  Returns:
    A flax optimizer.
  """
  if name == 'adam':
    logging.info('Creating Adam optimizer with settings lr=%f, beta1=%f, '
                 'beta2=%f, eps=%f', learning_rate, beta1, beta2, eps)
    return optim.Adam(
        learning_rate=learning_rate, beta1=beta1, beta2=beta2, eps=eps)
  elif name == 'rmsprop':
    logging.info('Creating RMSProp optimizer with settings lr=%f, beta2=%f, '
                 'eps=%f', learning_rate, beta2, eps)
    return optim.RMSProp(learning_rate=learning_rate, beta2=beta2, eps=eps,
                         centered=centered)
  else:
    raise ValueError('Unsupported optimizer {}'.format(name))


def huber_loss(targets, predictions, delta=1.0):
  x = jnp.abs(targets - predictions)
  return jnp.where(x <= delta,
                   0.5 * x**2,
                   0.5 * delta**2 + delta * (x - delta))


@functools.partial(jax.jit, static_argnums=(7))
def train(target_network, optimizer, states, actions, next_states, rewards,
          terminals, cumulative_gamma):
  """Run the training step."""
  def loss_fn(model, target):
    q_values = jax.vmap(model, in_axes=(0))(states).q_values
    q_values = jnp.squeeze(q_values)
    replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
    loss = jnp.mean(jax.vmap(huber_loss)(target, replay_chosen_q))
    return loss
  grad_fn = jax.value_and_grad(loss_fn)
  target = target_q(target_network,
                    next_states,
                    rewards,
                    terminals,
                    cumulative_gamma)
  loss, grad = grad_fn(optimizer.target, target)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss


def target_q(target_network, next_states, rewards, terminals, cumulative_gamma):
  """Compute the target Q-value."""
  q_vals = jax.vmap(target_network, in_axes=(0))(next_states).q_values
  q_vals = jnp.squeeze(q_vals)
  replay_next_qt_max = jnp.max(q_vals, 1)
  # Calculate the Bellman target value.
  #   Q_t = R_t + \gamma^N * Q'_t+1
  # where,
  #   Q'_t+1 = \argmax_a Q(S_t+1, a)
  #          (or) 0 if S_t is a terminal state,
  # and
  #   N is the update horizon (by default, N=1).
  return jax.lax.stop_gradient(rewards + cumulative_gamma * replay_next_qt_max *
                               (1. - terminals))


@gin.configurable
@functools.partial(jax.jit, static_argnums=(0, 2, 3))
def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
  """Returns the current epsilon for the agent's epsilon-greedy policy.

  This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
  al., 2015). The schedule is as follows:
    Begin at 1. until warmup_steps steps have been taken; then
    Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
    Use epsilon from there on.

  Args:
    decay_period: float, the period over which epsilon is decayed.
    step: int, the number of training steps completed so far.
    warmup_steps: int, the number of steps taken before epsilon is decayed.
    epsilon: float, the final value to which to decay the epsilon parameter.

  Returns:
    A float, the current epsilon value computed according to the schedule.
  """
  steps_left = decay_period + warmup_steps - step
  bonus = (1.0 - epsilon) * steps_left / decay_period
  bonus = jnp.clip(bonus, 0., 1. - epsilon)
  return epsilon + bonus


@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 9, 10))
def select_action(network, state, rng, num_actions, eval_mode,
                  epsilon_eval, epsilon_train, epsilon_decay_period,
                  training_steps, min_replay_history, epsilon_fn):
  """Select an action from the set of available actions.

  Chooses an action randomly with probability self._calculate_epsilon(), and
  otherwise acts greedily according to the current Q-value estimates.

  Args:
    network: Jax Module to use for inference.
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

  Returns:
    rng: Jax random number generator.
    action: int, the selected action.
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
                        jnp.argmax(network(state).q_values, axis=1)[0])


@gin.configurable
class JaxDQNAgent(object):
  """A JAX implementation of the DQN agent."""

  def __init__(self,
               num_actions,
               observation_shape=NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=NATURE_DQN_DTYPE,
               stack_size=NATURE_DQN_STACK_SIZE,
               network=networks.NatureDQNNetwork,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               eval_mode=False,
               max_tf_checkpoints_to_keep=4,
               optimizer='adam',
               summary_writer=None,
               summary_writing_frequency=500,
               allow_partial_reload=False,
               seed=None):
    """Initializes the agent and constructs the necessary components.

    Note: We are using the Adam optimizer by default for JaxDQN, which differs
          from the original NatureDQN and the dopamine TensorFlow version. In
          the experiments we have ran, we have found that using Adam yields
          improved training performance.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints describing the observation shape.
      observation_dtype: jnp.dtype, specifies the type of the observations.
      stack_size: int, number of frames to use in state stack.
      network: Jax network to use for training.
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
      eval_mode: bool, True for evaluation and False for training.
      max_tf_checkpoints_to_keep: int, the number of TensorFlow checkpoints to
        keep.
      optimizer: str, name of optimizer to use.
      summary_writer: SummaryWriter object for outputting training statistics.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      allow_partial_reload: bool, whether we allow reloading a partial agent
        (for instance, only the network parameters).
      seed: int, a seed for DQN's internal RNG, used for initialization and
        sampling actions. If None, will use the current time in nanoseconds.
    """
    assert isinstance(observation_shape, tuple)
    seed = int(time.time() * 1e6) if seed is None else seed
    logging.info('Creating %s agent with the following parameters:',
                 self.__class__.__name__)
    logging.info('\t gamma: %f', gamma)
    logging.info('\t update_horizon: %f', update_horizon)
    logging.info('\t min_replay_history: %d', min_replay_history)
    logging.info('\t update_period: %d', update_period)
    logging.info('\t target_update_period: %d', target_update_period)
    logging.info('\t epsilon_train: %f', epsilon_train)
    logging.info('\t epsilon_eval: %f', epsilon_eval)
    logging.info('\t epsilon_decay_period: %d', epsilon_decay_period)
    logging.info('\t optimizer: %s', optimizer)
    logging.info('\t max_tf_checkpoints_to_keep: %d',
                 max_tf_checkpoints_to_keep)
    logging.info('\t seed: %d', seed)

    self.num_actions = num_actions
    self.observation_shape = tuple(observation_shape)
    self.observation_dtype = observation_dtype
    self.stack_size = stack_size
    self.network = network.partial(num_actions=num_actions)
    self.gamma = gamma
    self.update_horizon = update_horizon
    self.cumulative_gamma = math.pow(gamma, update_horizon)
    self.min_replay_history = min_replay_history
    self.target_update_period = target_update_period
    self.epsilon_fn = epsilon_fn
    self.epsilon_train = epsilon_train
    self.epsilon_eval = epsilon_eval
    self.epsilon_decay_period = epsilon_decay_period
    self.update_period = update_period
    self.eval_mode = eval_mode
    self.training_steps = 0
    self.summary_writer = summary_writer
    self.summary_writing_frequency = summary_writing_frequency
    self.allow_partial_reload = allow_partial_reload

    self._rng = jax.random.PRNGKey(seed)
    state_shape = self.observation_shape + (stack_size,)
    self.state = onp.zeros(state_shape)
    self._replay = self._build_replay_buffer()
    self._optimizer_name = optimizer
    self._build_networks_and_optimizer()

    # Variables to be initialized by the agent once it interacts with the
    # environment.
    self._observation = None
    self._last_observation = None

  def _create_network(self, name):
    """Builds the convolutional network used to compute the agent's Q-values.

    Args:
      name: str, this name is passed to the Jax Module.

    Returns:
      network: Jax Model, the network instantiated by Jax.
    """
    _, initial_params = self.network.init(self._rng, name=name,
                                          x=self.state,
                                          num_actions=self.num_actions)
    return nn.Model(self.network, initial_params)

  def _build_networks_and_optimizer(self):
    online_network = self._create_network(name='Online')
    optimizer_def = create_optimizer(self._optimizer_name)
    self.optimizer = optimizer_def.create(online_network)
    self.target_network = self._create_network(name='Target')

  @property
  def online_network(self):
    # We set up this symlink to use the model that is being updated by the
    # optimizer.
    return self.optimizer.target

  def _build_replay_buffer(self):
    """Creates the replay buffer used by the agent."""
    return circular_replay_buffer.OutOfGraphReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype)

  def _sample_from_replay_buffer(self):
    samples = self._replay.sample_transition_batch()
    types = self._replay.get_transition_elements()
    self.replay_elements = collections.OrderedDict()
    for element, element_type in zip(samples, types):
      self.replay_elements[element_type.name] = element

  def _sync_weights(self):
    """Syncs the target_network weights with the online_network weights."""
    self.target_network = self.target_network.replace(
        params=self.online_network.params)

  def _reset_state(self):
    """Resets the agent state by filling it with zeros."""
    self.state.fill(0)

  def _record_observation(self, observation):
    """Records an observation and update state.

    Extracts a frame from the observation vector and overwrites the oldest
    frame in the state buffer.

    Args:
      observation: numpy array, an observation from the environment.
    """
    # Set current observation. We do the reshaping to handle environments
    # without frame stacking.
    self._observation = onp.reshape(observation, self.observation_shape)
    # Swap out the oldest frame with the current frame.
    self.state = onp.roll(self.state, -1, axis=-1)
    self.state[..., -1] = self._observation

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

    self._rng, self.action = select_action(self.online_network,
                                           self.state,
                                           self._rng,
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

    self._rng, self.action = select_action(self.online_network,
                                           self.state,
                                           self._rng,
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

  def end_episode(self, reward, terminal=True):
    """Signals the end of the episode to the agent.

    We store the observation of the current time step, which is the last
    observation of the episode.

    Args:
      reward: float, the last reward from the environment.
      terminal: bool, whether the last state-action led to a terminal state.
    """
    if not self.eval_mode:
      argspec = inspect.getfullargspec(self._store_transition)
      if 'episode_end' in argspec.args or 'episode_end' in argspec.kwonlyargs:
        self._store_transition(
            self._observation, self.action, reward, terminal, episode_end=True)
      else:
        logging.warning(
            '_store_transition function doesn\'t have episode_end arg.')
        self._store_transition(self._observation, self.action, reward, terminal)

  def _train_step(self):
    """Runs a single training step.

    Runs training if both:
      (1) A minimum number of frames have been added to the replay buffer.
      (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online_network to target_network if training steps
    is a multiple of target update period.
    """
    # Run a train op at the rate of self.update_period if enough training steps
    # have been run. This matches the Nature DQN behaviour.
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()
        self.optimizer, loss = train(self.target_network,
                                     self.optimizer,
                                     self.replay_elements['state'],
                                     self.replay_elements['action'],
                                     self.replay_elements['next_state'],
                                     self.replay_elements['reward'],
                                     self.replay_elements['terminal'],
                                     self.cumulative_gamma)
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          summary = tf.compat.v1.Summary(value=[
              tf.compat.v1.Summary.Value(tag='HuberLoss', simple_value=loss)])
          self.summary_writer.add_summary(summary, self.training_steps)
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1

  def _store_transition(self,
                        last_observation,
                        action,
                        reward,
                        is_terminal,
                        *args,
                        priority=None,
                        episode_end=False):
    """Stores a transition when in training mode.

    Stores the following tuple in the replay buffer (last_observation, action,
    reward, is_terminal, priority).

    Args:
      last_observation: Last observation, type determined via observation_type
        parameter in the replay_memory constructor.
      action: An integer, the action taken.
      reward: A float, the reward.
      is_terminal: Boolean indicating if the current state is a terminal state.
      *args: Any, other items to be added to the replay buffer.
      priority: Float. Priority of sampling the transition. If None, the default
        priority will be used. If replay scheme is uniform, the default priority
        is 1. If the replay scheme is prioritized, the default priority is the
        maximum ever seen [Schaul et al., 2015].
      episode_end: bool, whether this transition is the last for the episode.
        This can be different than terminal when ending the episode because
        of a timeout, for example.
    """
    is_prioritized = isinstance(
        self._replay,
        prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer)
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
        'online_params': self.online_network.params,
        'target_params': self.target_network.params
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
      bundle_dictionary: dict, containing additional Python objects owned by
        the agent.

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
      online_network = self.online_network.replace(
          params=bundle_dictionary['online_params'])
      # We recreate the optimizer with the new online weights.
      optimizer_def = create_optimizer(self._optimizer_name)
      self.optimizer = optimizer_def.create(online_network)
      self.target_network = self.target_network.replace(
          params=bundle_dictionary['target_params'])
    elif not self.allow_partial_reload:
      return False
    else:
      logging.warning("Unable to reload the agent's parameters!")
    return True

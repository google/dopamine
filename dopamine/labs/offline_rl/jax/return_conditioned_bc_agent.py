# coding=utf-8
# Copyright 2023 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compact implementation of return-conditioned BC agent in JAX."""

import functools

from absl import logging
from dopamine.jax import losses
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.agents.rainbow import rainbow_agent
from dopamine.labs.offline_rl.jax import offline_rainbow_agent
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tensorflow as tf

c51_loss_fn = jax.vmap(losses.softmax_cross_entropy_loss_with_logits)
huber_loss_fn = jax.vmap(losses.huber_loss)


@functools.partial(jax.vmap, in_axes=(None, 0, 0))
def get_q_values(model, states, returns_to_condition):
  outputs = model(states, return_to_condition=returns_to_condition)
  return jnp.squeeze(outputs.q_values)


@functools.partial(
    jax.jit,
    static_argnames=('network_def', 'optimizer'))
def train(
    network_def, online_params, optimizer, optimizer_state,
    states, actions, returns_to_condition, support):
  """Run a training step."""

  def loss_fn(params):
    """Computes the distributional loss for C51 or huber loss for DQN."""

    def q_func(state, return_to_condition):
      return network_def.apply(
          params,
          state,
          support=support,
          return_to_condition=return_to_condition,
      )
    q_values = get_q_values(q_func, states, returns_to_condition)
    replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
    bc_loss = jnp.mean(
        jax.scipy.special.logsumexp(q_values, axis=-1) - replay_chosen_q)
    return bc_loss

  grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
  # Get the unweighted loss without taking its mean for updating priorities.
  # outputs[1] correspond to the per-example TD loss.
  loss, grad = grad_fn(online_params)
  updates, optimizer_state = optimizer.update(
      grad, optimizer_state, params=online_params)
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, loss


class WrappedNetworkDef(object):
  """Wrapping network def for a specified return_to_condition."""

  def __init__(self, network_def, min_return, max_return):
    self.network_def = network_def
    self._min_return = min_return
    self._max_return = max_return
    self.set_return_to_condition()

  def set_return_to_condition(self, return_multiplier=1.0):
    return_to_condition = (
        self._max_return - self._min_return
    ) * return_multiplier + self._max_return
    self._return_to_condition = jnp.array(
        return_to_condition, dtype=jnp.float32)

  def apply(self, params, x, support, key=None):
    del key
    return self.network_def.apply(
        params,
        x,
        support=support,
        return_to_condition=self._return_to_condition,
    )


@gin.configurable
class JaxReturnConditionedBCAgent(
    offline_rainbow_agent.OfflineJaxRainbowAgent):
  """Return conditioned Behavior cloning agent."""

  def __init__(self,
               num_actions,
               replay_data_dir,
               summary_writer=None,
               use_tfds=True):
    """Initializes the agent and constructs the necessary components.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      replay_data_dir: str, log Directory from which to load the replay buffer.
      summary_writer: SummaryWriter object for outputting training statistics.
      use_tfds: Whether to use tfds replay buffer.
    """

    logging.info('Creating JaxReturnConditionedBCAgent ..')
    super().__init__(
        num_actions,
        replay_data_dir=replay_data_dir,
        bc_coefficient=1.0,
        td_coefficient=0.0,
        use_tfds=use_tfds,
        summary_writer=summary_writer)
    self._create_wrapped_network()

  def _training_step_update(self):
    """Runs a single training step."""
    self._sample_from_replay_buffer()
    states = self.preprocess_fn(self.replay_elements['state'])

    (self.optimizer_state, self.online_params, bc_loss) = train(
        self.network_def,
        self.online_params,
        self.optimizer,
        self.optimizer_state,
        states,
        self.replay_elements['action'],
        self.replay_elements['episode_return'],
        self._support,
    )

    if (self.training_steps > 0 and
        self.training_steps % self.summary_writing_frequency == 0):
      if self.summary_writer is not None:
        with self.summary_writer.as_default():
          tf.summary.scalar('Losses/BCLoss', bc_loss,
                            step=self.training_steps)
        self.summary_writer.flush()
    if self._use_tfds:
      self.log_gradient_steps_per_epoch()

    self.training_steps += 1

  def _build_networks_and_optimizer(self):
    self._rng, rng = jax.random.split(self._rng)
    self.online_params = self.network_def.init(
        rng, x=self.state, support=self._support,
        return_to_condition=jnp.array(0.0, dtype=jnp.float32))
    self.optimizer = dqn_agent.create_optimizer(self._optimizer_name)
    self.optimizer_state = self.optimizer.init(self.online_params)
    self.target_network_params = self.online_params

  def _create_wrapped_network(self):
    min_return, max_return = self._replay.min_max_returns
    self.wrapped_network_def = WrappedNetworkDef(
        self.network_def, min_return, max_return)

  def set_return_to_condition(self, return_multiplier):
    self.wrapped_network_def.set_return_to_condition(return_multiplier)

  def step(self, reward, observation):
    """Returns the agent's next action and update agent's state.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    self._record_observation(observation)
    return self._get_action()

  def _get_action(self):
    state = self.preprocess_fn(self.state)
    self._rng, self.action = rainbow_agent.select_action(
        self.wrapped_network_def, self.online_params, state, self._rng,
        self.num_actions, self.eval_mode, self.epsilon_eval, self.epsilon_train,
        self.epsilon_decay_period, self.training_steps, self.min_replay_history,
        self.epsilon_fn, self._support)
    self.action = onp.asarray(self.action)
    return self.action

  def begin_episode(self, observation):
    """Returns the agent's first action for this episode."""
    self._reset_state()
    self._record_observation(observation)
    if not self.eval_mode:
      self._train_step()
    return self._get_action()

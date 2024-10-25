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
"""PPO Agent for CALE."""

import functools

from absl import logging
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.agents.ppo import ppo_agent
from dopamine.labs.cale import utils
import gin
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf


@functools.partial(
    jax.jit,
    static_argnames=[
        'network_def',
        'epsilon_decay_period',
        'min_replay_history',
        'epsilon_eval',
        'epsilon_train',
        'action_shape',
    ],
)
def select_action_eps_greedy(
    network_def,
    params,
    state,
    rng,
    training_steps,
    min_replay_history,
    epsilon_eval,
    epsilon_train,
    epsilon_decay_period,
    action_shape,
    eval_mode=False,
):
  """Epsilon-greedy action selection."""
  epsilon = jnp.where(
      eval_mode,
      epsilon_eval,
      dqn_agent.linearly_decaying_epsilon(
          epsilon_decay_period,
          training_steps,
          min_replay_history,
          epsilon_train,
      ),
  )
  rng, rng1, rng2, rng3 = jax.random.split(rng, num=4)
  p = jax.random.uniform(rng1)
  return rng, jnp.where(
      p <= epsilon,
      jax.random.uniform(rng2, action_shape),
      ppo_agent.select_action(network_def, params, state, rng3)[1],
  )


@gin.configurable
class PPOCALEAgent(ppo_agent.PPOAgent):
  """PPO Agent for CALE."""

  def __init__(
      self,
      action_shape,
      action_limits,
      observation_shape,
      epsilon_train=0.01,
      epsilon_eval=0.001,
      epsilon_decay_period=250_000,
      exploration_strategy='standard',
      log_action_distributions=True,
      summary_writer=None,
  ):
    """Initialize agent."""
    super().__init__(
        action_shape=action_shape,
        action_limits=action_limits,
        observation_shape=observation_shape,
        summary_writer=summary_writer,
    )
    logging.info('\t epsilon_train: %f', epsilon_train)
    logging.info('\t epsilon_eval: %f', epsilon_eval)
    logging.info('\t epsilon_decay_period: %d', epsilon_decay_period)
    logging.info('\t exploration_strategy: %s', exploration_strategy)
    self.epsilon_train = epsilon_train
    self.epsilon_eval = epsilon_eval
    self.epsilon_decay_period = epsilon_decay_period
    self.exploration_strategy = exploration_strategy
    self.action_distributions = None
    if log_action_distributions:
      self.action_distributions = np.zeros((18,))

  def _select_action(self):
    self._rng, action = select_action_eps_greedy(
        self.network_def,
        self.network_params,
        self.state,
        self._rng,
        self.training_steps,
        self.min_replay_history,
        self.epsilon_eval,
        self.epsilon_train,
        self.epsilon_decay_period,
        self.action_shape,
        eval_mode=self.eval_mode,
    )
    self.action = np.asarray(action)
    return self.action

  def _maybe_log_action_distribution(self):
    if self.action_distributions is not None:
      action_int = utils.get_action_number(
          self.action[0], self.action[1], self.action[2]
      )
      self.action_distributions[action_int] += 1

  def begin_episode(self, observation):
    if self.exploration_strategy == 'standard':
      self.action = super().begin_episode(observation)
    else:
      self._reset_state()
      self._record_observation(observation)
      if not self.eval_mode:
        self._train_step()

      self.action = self._select_action()

    self._maybe_log_action_distribution()
    return self.action

  def step(self, reward, observation):
    if self.exploration_strategy == 'standard':
      self.action = super().step(reward, observation)
    else:
      self._last_observation = self._observation
      self._record_observation(observation)

      if not self.eval_mode:
        self._store_transition(
            self._last_observation, self.action, reward, False
        )
        self._train_step()

      self.action = self._select_action()

    self._maybe_log_action_distribution()
    return self.action

  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    if self.action_distributions is not None:
      filename = f'{checkpoint_dir}/action_distributions_{iteration_number}.npy'
      with tf.io.gfile.GFile(filename, 'w') as f:
        np.save(f, self.action_distributions)

    return super().bundle_and_checkpoint(checkpoint_dir, iteration_number)

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
"""DQN agent that uses MoE networks and logs expert weight histograms."""

import functools
from absl import logging
from dopamine.jax import losses
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.labs.moes.agents import losses as moe_losses
from dopamine.labs.moes.agents import types
from dopamine.labs.moes.architectures import networks
from dopamine.labs.moes.architectures import types as arch_types
from dopamine.metrics import statistics_instance
import gin
import jax
import jax.numpy as jnp
import numpy as np
import optax


@functools.partial(
    jax.jit,
    static_argnames=(
        'network_def',
        'optimizer',
        'cumulative_gamma',
        'log_moe_statistics',
        'log_dormant_statistics',
        'loss_type',
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
    cumulative_gamma,
    rng,
    log_moe_statistics,
    log_dormant_statistics,
    loss_type='mse',
):
  """Run the training step."""
  batch_size = states.shape[0]
  rng1, rng2, rng3, rng4 = jax.random.split(rng, num=4)

  def loss_fn(params, target):
    def q_online(state, net_rng):
      return network_def.apply(params, state, key=net_rng)

    rng_online = jnp.stack(jax.random.split(rng1, num=batch_size))
    net_outputs = jax.vmap(q_online)(states, net_rng=rng_online)
    q_values = jnp.squeeze(net_outputs.q_values)
    replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
    if loss_type == 'huber':
      td_loss = jnp.mean(jax.vmap(losses.huber_loss)(target, replay_chosen_q))
    else:
      td_loss = jnp.mean(jax.vmap(losses.mse_loss)(target, replay_chosen_q))
    aux_vars = {}
    aux_loss_values = 0.0
    if isinstance(net_outputs, arch_types.MoENetworkReturn):
      aux_losses = moe_losses.aux_loss(
          types.MoELossParameters(
              moe_out=net_outputs.moe_out,
              num_experts=network_def.num_experts,
              num_selected_experts=network_def.num_selected_experts,
              key=rng3,
          )
      )
      aux_loss_values = jnp.sum(
          jnp.array([aux_loss.value for aux_loss in aux_losses])
      )
      aux_vars.update({
          'top_experts': net_outputs.moe_out.router_out.top_experts,
          'experts_prob': net_outputs.moe_out.router_out.probabilities,
          'aux_losses': aux_losses,
          'hidden_act': net_outputs.moe_out.experts_hidden,
      })
    elif isinstance(net_outputs, arch_types.BaselineNetworkReturn):
      aux_vars.update({'hidden_act': net_outputs.hidden_act})

    loss = td_loss + aux_loss_values
    aux_vars.update({'combined_loss': loss, 'td_loss': td_loss})
    return loss, aux_vars

  def q_target(state, net_rng):
    return network_def.apply(target_params, state, key=net_rng)

  target = target_q(
      q_target, next_states, rewards, terminals, cumulative_gamma, rng2
  )
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, aux_vars), grad = grad_fn(online_params, target)
  updates, optimizer_state = optimizer.update(
      grad, optimizer_state, params=online_params
  )
  online_params = optax.apply_updates(online_params, updates)

  train_returns = {
      'optimizer_state': optimizer_state,
      'online_params': online_params,
      'rng': rng4,
  }
  train_returns.update(aux_vars)

  # log some statistics of MoE for debugging.
  if (
      log_moe_statistics
      and networks.MoEType[network_def.moe_type] != networks.MoEType.BASELINE
      and (
          networks.MoEType[network_def.moe_type]
          != networks.MoEType.SIMPLICIAL_EMBEDDING_V1
      )
      and (
          networks.MoEType[network_def.moe_type]
          != networks.MoEType.SIMPLICIAL_EMBEDDING_V2
      )
  ):
    moe_statistics = {}
    experts_prob = jnp.mean(jnp.mean(aux_vars['experts_prob'], axis=0), axis=0)
    # TODO(gsokar) revisit this if we explore multiple routers.

    if networks.MoEType[network_def.moe_type] == networks.MoEType.SOFTMOE:
      grads_router = [
          grad['params']['SoftMoE_0']['phi_weights'],
          online_params['params']['SoftMoE_0']['phi_weights'],
      ]
    else:
      grads_router = [
          grad['params']['MoE_0']['router']['Dense_0']['kernel'],
          online_params['params']['MoE_0']['router']['Dense_0']['kernel'],
      ]
    mean_abs_grad_router = jnp.mean(jnp.abs(grads_router[0]))
    mean_abs_weights_router = jnp.mean(jnp.abs(grads_router[1]))

    moe_statistics['Stats/mean_abs_grad_router'] = mean_abs_grad_router
    moe_statistics['Stats/mean_abs_weights_router'] = mean_abs_weights_router
    top_experts_one_hot = jax.nn.one_hot(
        aux_vars['top_experts'], network_def.num_experts
    )
    avg_top_experts_one_hot = jnp.mean(
        jnp.sum(top_experts_one_hot, axis=[1, 2]), axis=0
    )
    for i in range(network_def.num_experts):
      moe_statistics[f'Stats/expert{i}gate'] = experts_prob[i]
      moe_statistics[f'Stats/selected_expert{i}'] = avg_top_experts_one_hot[i]
      if networks.MoEType[network_def.moe_type] == networks.MoEType.DROPLESSMOE:
        # TODO(obandoceron): 'DROPLESSMOE' has not support yet.
        mean_abs_grad = jnp.mean(jnp.abs(grad['params']['MoE_0']['weights'][i]))
      elif networks.MoEType[network_def.moe_type] == networks.MoEType.SOFTMOE:
        mean_abs_grad = jnp.mean(
            jnp.abs(grad['params']['SoftMoE_0']['phi_weights'][:, i])
        )
      else:
        mean_abs_grad = jnp.mean(
            jnp.abs(grad['params']['MoE_0']['router']['Dense_0']['kernel'][i])
        )
      moe_statistics[f'Stats/mean_abs_grad_expert{i}'] = mean_abs_grad
    train_returns.update(moe_statistics)

  # log statistics for dormant neurons in the expert hidden layer.
  if log_dormant_statistics and (
      networks.MoEType[network_def.moe_type] != networks.MoEType.DROPLESSMOE
  ):
    dormancy_threshold = 0.0
    dormancy_statistics = {}
    # average over batch
    score = jnp.mean(jnp.abs(aux_vars['hidden_act']), axis=(0))
    # average over tokens
    if (
        networks.MoEType[network_def.moe_type] != networks.MoEType.BASELINE
        and (
            networks.MoEType[network_def.moe_type]
            != networks.MoEType.SIMPLICIAL_EMBEDDING_V1
        )
        and (
            networks.MoEType[network_def.moe_type]
            != networks.MoEType.SIMPLICIAL_EMBEDDING_V2
        )
    ):
      score = jnp.mean(score, axis=0)
    # normalized score over layer
    score = score / (jnp.sum(score) + 1e-9)
    dormancy_statistics['Stats/dormant_percentage'] = (
        (jnp.count_nonzero(score <= dormancy_threshold)).astype(float)
        / jnp.size(score)
    ) * 100.0
    train_returns.update(dormancy_statistics)
  return train_returns


# We are overriding the following function as we need to pass in the rng to the
# network inference calls.
def target_q(
    target_network, next_states, rewards, terminals, cumulative_gamma, rng
):
  """Compute the target Q-value."""
  batch_size = next_states.shape[0]
  rng_target = jnp.stack(jax.random.split(rng, num=batch_size))
  q_vals = jax.vmap(target_network)(next_states, rng_target).q_values
  q_vals = jnp.squeeze(q_vals)
  replay_next_qt_max = jnp.max(q_vals, 1)
  # Calculate the Bellman target value.
  #   Q_t = R_t + \gamma^N * Q'_t+1
  # where,
  #   Q'_t+1 = \argmax_a Q(S_t+1, a)
  #          (or) 0 if S_t is a terminal state,
  # and
  #   N is the update horizon (by default, N=1).
  return jax.lax.stop_gradient(
      rewards + cumulative_gamma * replay_next_qt_max * (1.0 - terminals)
  )


# We are overriding the following function as we need to pass in the rng to the
# network inference calls.
@functools.partial(
    jax.jit,
    static_argnames=(
        'network_def',
        'num_actions',
        'eval_mode',
        'epsilon_eval',
        'epsilon_train',
        'epsilon_decay_period',
        'min_replay_history',
        'epsilon_fn',
    ),
)
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
  return rng, jnp.where(
      p <= epsilon,
      jax.random.randint(rng2, (), 0, num_actions),
      jnp.argmax(network_def.apply(params, state, key=rng3).q_values),
  )


@gin.configurable
class DQNMoEAgent(dqn_agent.JaxDQNAgent):
  """DQN agent that uses MoE networks."""

  def __init__(
      self,
      num_actions,
      summary_writer=None,
      log_moe_statistics=True,
      log_dormant_statistics=True,
      num_updates_per_train_step=1,
  ):
    super().__init__(num_actions, summary_writer=summary_writer)
    self.log_moe_statistics = log_moe_statistics
    self.log_dormant_statistics = log_dormant_statistics
    self._num_updates_per_train_step = num_updates_per_train_step
    logging.info('\t Creating %s ...', self.__class__.__name__)
    logging.info('\t log_moe_statistics: %s', self.log_moe_statistics)
    logging.info('\t log_dormant_statistics %s', self.log_dormant_statistics)
    logging.info(
        '\t num_updates_per_train_step: %s', self._num_updates_per_train_step
    )

  # We are overriding the following function as we need to pass in the rng to
  # the network inference calls.
  def _build_networks_and_optimizer(self):
    self._rng, rng = jax.random.split(self._rng)
    state = self.preprocess_fn(self.state)
    self.online_params = self.network_def.init(rng, x=state, key=rng)
    self.optimizer = dqn_agent.create_optimizer(self._optimizer_name)
    self.optimizer_state = self.optimizer.init(self.online_params)
    self.target_network_params = self.online_params

  # We are overriding the following function as we need to pass in the rng to
  # the network inference calls.
  def begin_episode(self, observation):
    """Returns the agent's first action for this episode."""
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
    )
    self.action = np.asarray(self.action)
    return self.action

  # We are overriding the following function as we need to pass in the rng to
  # the network inference calls.
  def step(self, reward, observation):
    """Records the most recent transition and returns the next action."""
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
    )
    self.action = np.asarray(self.action)
    return self.action

  def _train_step(self):
    """Runs a single training step."""
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        for _ in range(self._num_updates_per_train_step):
          self._training_step_update()

      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1

  def _training_step_update(self):
    # We are using # gradient_update_steps in our calculations and logging.
    self._sample_from_replay_buffer()
    states = self.preprocess_fn(self.replay_elements['state'])
    next_states = self.preprocess_fn(self.replay_elements['next_state'])
    train_returns = train(
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
        self.cumulative_gamma,
        self._rng,
        self.log_moe_statistics,
        self.log_dormant_statistics,
        self._loss_type,
    )

    self.optimizer_state = train_returns['optimizer_state']
    self.online_params = train_returns['online_params']
    self._rng = train_returns['rng']
    loss = train_returns['combined_loss']
    td_loss = train_returns['td_loss']

    if (
        self.summary_writer is not None
        and self.training_steps > 0
        and self.training_steps % self.summary_writing_frequency == 0
    ):
      if hasattr(self, 'collector_dispatcher'):
        statistics = [
            statistics_instance.StatisticsInstance(
                'CombinedLoss', np.asarray(loss), step=self.training_steps
            ),
            statistics_instance.StatisticsInstance(
                'TDLoss', np.asarray(td_loss), step=self.training_steps
            ),
        ]
        # Add any extra statistics returned.
        if 'aux_losses' in train_returns:
          for aux_loss in train_returns['aux_losses']:
            for aux_stat in aux_loss.statistics:
              statistics.append(
                  statistics_instance.StatisticsInstance(
                      types.ID_TO_NAME[int(aux_stat.name_id)],
                      np.asarray(aux_stat.value),
                      type=types.ID_TO_TYPE[int(aux_stat.type_id)],
                      step=self.training_steps,
                  )
              )

        # Add any moe or dormant neuron statistics returned.
        for k, v in train_returns.items():
          if k.startswith('Stats'):
            statistics.append(
                statistics_instance.StatisticsInstance(
                    k, np.asarray(v), type='scalar', step=self.training_steps
                )
            )

        self.collector_dispatcher.write(
            statistics, collector_allowlist=self._collector_allowlist
        )

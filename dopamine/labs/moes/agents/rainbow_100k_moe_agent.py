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
"""Atari 100k rainbow agent with MoE modules."""

import functools
from absl import logging
from dopamine.jax import losses
from dopamine.jax.agents.full_rainbow import full_rainbow_agent
from dopamine.labs.atari_100k import atari_100k_rainbow_agent as base_100k_rainbow
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
        'double_dqn',
        'distributional',
        'mse_loss',
        'log_moe_statistics',
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
    log_moe_statistics,
):
  """Run a training step."""

  batch_size = states.shape[0]
  # Split the current rng into 4 for updating the rng after this call
  rng, rng1, rng2, rng3 = jax.random.split(rng, num=4)

  def q_online(state, key):
    return network_def.apply(online_params, state, key=key, support=support)

  def q_target(state, key):
    return network_def.apply(target_params, state, key=key, support=support)

  def loss_fn(params, target, loss_multipliers):
    """Computes the distributional loss for C51 or huber loss for DQN."""

    def q_online(state, key):
      return network_def.apply(params, state, key=key, support=support)

    rng_online = jnp.stack(jax.random.split(rng1, num=batch_size))
    net_outputs = jax.vmap(q_online)(states, key=rng_online)
    if distributional:
      logits = net_outputs.logits
      logits = jnp.squeeze(logits)
      # Fetch the logits for its selected action. We use vmap to perform this
      # indexing across the batch.
      chosen_action_logits = jax.vmap(lambda x, y: x[y])(logits, actions)
      td_loss = jax.vmap(losses.softmax_cross_entropy_loss_with_logits)(
          target, chosen_action_logits
      )
    else:
      q_values = net_outputs.q_values
      q_values = jnp.squeeze(q_values)
      replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)

      td_loss_fn = losses.mse_loss if mse_loss else losses.huber_loss
      td_loss = jax.vmap(td_loss_fn)(target, replay_chosen_q)

    aux_vars = {}
    aux_loss_values = 0.0
    mean_td_loss = jnp.mean(loss_multipliers * td_loss)
    aux_losses = []
    if isinstance(net_outputs, arch_types.MoENetworkReturn):
      # We may be running a BASELINE agent, which would not contain any MoE
      # statistics, so we condition this code on *not* being a BASELINE..
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
      })

    loss = mean_td_loss + aux_loss_values
    aux_vars.update({
        'combined_loss': loss,
        'mean_td_loss': mean_td_loss,
        'td_loss': td_loss,
        'aux_losses': aux_losses,
    })
    return loss, aux_vars

  # Use the weighted mean loss for gradient computation.
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  target = full_rainbow_agent.target_output(
      q_online,
      q_target,
      next_states,
      rewards,
      terminals,
      support,
      cumulative_gamma,
      double_dqn,
      distributional,
      rng2,
  )

  # Get the unweighted loss without taking its mean for updating priorities.
  (_, aux_vars), grad = grad_fn(online_params, target, loss_weights)
  updates, optimizer_state = optimizer.update(
      grad, optimizer_state, params=online_params
  )
  online_params = optax.apply_updates(online_params, updates)
  train_returns = {
      'optimizer_state': optimizer_state,
      'online_params': online_params,
      'rng': rng,
  }
  train_returns.update(aux_vars)
  # TODO(psc): Refactor this so we're not duplicating this code with DQN.
  # log some statistics of MoE for debugging.
  if (
      log_moe_statistics
      and networks.MoEType[network_def.moe_type] != networks.MoEType.BASELINE
      and networks.MoEType[network_def.moe_type] != networks.MoEType.SOFTMOE
  ):
    moe_statistics = {}
    experts_prob = jnp.mean(jnp.mean(aux_vars['experts_prob'], axis=0), axis=0)
    # TODO(gsokar) revisit this if we explore multiple routers.
    mean_abs_grad_router = jnp.mean(
        jnp.abs(grad['params']['MoE_0']['router']['Dense_0']['kernel'])
    )
    mean_abs_weights_router = jnp.mean(
        jnp.abs(online_params['params']['MoE_0']['router']['Dense_0']['kernel'])
    )
    moe_statistics['MoE_Stats/mean_abs_grad_router'] = mean_abs_grad_router
    moe_statistics['MoE_Stats/mean_abs_weights_router'] = (
        mean_abs_weights_router
    )
    top_experts_one_hot = jax.nn.one_hot(
        aux_vars['top_experts'], network_def.num_experts
    )
    avg_top_experts_one_hot = jnp.mean(
        jnp.sum(top_experts_one_hot, axis=[1, 2]), axis=0
    )
    for i in range(network_def.num_experts):
      moe_statistics[f'MoE_Stats/expert{i}gate'] = experts_prob[i]
      moe_statistics[f'MoE_Stats/selected_expert{i}'] = avg_top_experts_one_hot[
          i
      ]
      if networks.MoEType[network_def.moe_type] == networks.MoEType.DROPLESSMOE:
        # TODO(obandoceron): 'DROPLESSMOE' has not support yet.
        mean_abs_grad = jnp.mean(jnp.abs(grad['params']['MoE_0']['weights'][i]))
      else:
        mean_abs_grad = jnp.mean(
            jnp.abs(grad['params']['MoE_0']['router']['Dense_0']['kernel'][i])
        )
      moe_statistics[f'MoE_Stats/mean_abs_grad_expert{i}'] = mean_abs_grad
    train_returns.update(moe_statistics)

  return train_returns


@gin.configurable
class Atari100kRainbowMoEAgent(base_100k_rainbow.Atari100kRainbowAgent):
  """A compact implementation of agents for Atari 100k with MoEs."""

  def __init__(
      self,
      num_actions,
      summary_writer=None,
      seed=None,
      log_moe_statistics=False,
  ):
    """Creates the Rainbow-based agent for Atari 100k benchmark with MoEs."""
    super().__init__(
        num_actions=num_actions, seed=seed, summary_writer=summary_writer
    )
    self.log_moe_statistics = log_moe_statistics
    logging.info('\t Creating %s ...', self.__class__.__name__)
    logging.info('\t log_moe_statistics: %s', self.log_moe_statistics)

  def _train_step(self):
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        for _ in range(self._num_updates_per_train_step):
          self._training_step_update()

      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1

  def _training_step_update(self):
    """Gradient update during every training step."""

    self._sample_from_replay_buffer()
    self._rng, rng1, rng2 = jax.random.split(self._rng, num=3)
    states = self.train_preprocess_fn(self.replay_elements['state'], rng=rng1)
    next_states = self.train_preprocess_fn(
        self.replay_elements['next_state'], rng=rng2
    )

    if self._replay_scheme == 'prioritized':
      probs = self.replay_elements['sampling_probabilities']
      # Weight the loss by the inverse priorities.
      loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
      loss_weights /= jnp.max(loss_weights)
    else:
      # Uniform weights if not using prioritized replay.
      loss_weights = jnp.ones(states.shape[0])

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
        loss_weights,
        self._support,
        self.cumulative_gamma,
        self._double_dqn,
        self._distributional,
        self._mse_loss,
        self._rng,
        self.log_moe_statistics,
    )

    self.optimizer_state = train_returns['optimizer_state']
    self.online_params = train_returns['online_params']
    self._rng = train_returns['rng']
    loss = train_returns['combined_loss']
    td_loss = train_returns['td_loss']
    mean_td_loss = train_returns['mean_td_loss']

    if self._replay_scheme == 'prioritized':
      # TODO(psc): We may want to explore setting priorities on combined loss.
      self._replay.set_priority(
          self.replay_elements['indices'], jnp.sqrt(td_loss + 1e-10)
      )

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
                'TDLoss', np.asarray(mean_td_loss), step=self.training_steps
            ),
        ]
        # Add any extra statistics returned.
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

        # Add any moe statistics returned.
        for k, v in train_returns.items():
          if k.startswith('MoE_Stats'):
            statistics.append(
                statistics_instance.StatisticsInstance(
                    k, np.asarray(v), type='scalar', step=self.training_steps
                )
            )

        self.collector_dispatcher.write(
            statistics, collector_allowlist=self._collector_allowlist
        )

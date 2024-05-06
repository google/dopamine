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
"""Compact implementation of the full Rainbow agent in JAX with MoE modules."""

import functools

from absl import logging
from dopamine.jax import losses
from dopamine.jax.agents.full_rainbow import full_rainbow_agent
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
        'log_network_statistics',
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
    log_network_statistics,
):
  """Run a training step."""
  batch_size = states.shape[0]
  # Split the current rng into 4 for updating the rng after this call
  rng, rng1, rng2, rng3 = jax.random.split(rng, num=4)

  def q_online(state, key):
    return network_def.apply(online_params, state, key=key, support=support)

  def q_target(state, key):
    return network_def.apply(target_params, state, key=key, support=support)

  def per_sample_loss_fn(params, target):
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
    return td_loss, net_outputs

  def loss_fn(params, target, loss_multipliers):
    td_loss, net_outputs = per_sample_loss_fn(params, target)
    aux_vars = {}
    aux_loss_values = 0.0
    mean_td_loss = jnp.mean(loss_multipliers * td_loss)
    aux_losses = []
    if isinstance(net_outputs, arch_types.MoENetworkReturn):
      # We may be running a BASELINE agent, which would not contain any MoE
      # statistics, so we condition this code on *not* being a BASELINE.
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
      elif networks.MoEType[network_def.moe_type] == networks.MoEType.SOFTMOE:
        mean_abs_grad = jnp.mean(
            jnp.abs(grad['params']['SoftMoE_0']['phi_weights'][:, i])
        )
      else:
        mean_abs_grad = jnp.mean(
            jnp.abs(grad['params']['MoE_0']['router']['Dense_0']['kernel'][i])
        )
      moe_statistics[f'MoE_Stats/mean_abs_grad_expert{i}'] = mean_abs_grad
    train_returns.update(moe_statistics)
  if log_network_statistics:
    # Feature data
    feature_map = lambda x: q_online(x, jax.random.PRNGKey(0)).hidden_act
    features = jax.vmap(feature_map)(states)
    singular_vals = jnp.linalg.svd(features)[1]
    feature_vars = jnp.var(features, axis=0)
    feature_norm = jnp.mean(jnp.linalg.norm(feature_vars, axis=-1))

    # Empirical NTK data
    jac_fn = jax.jacrev(per_sample_loss_fn, has_aux=True)
    jacs = jac_fn(online_params, target)
    batch_product_fn = (
        lambda x: x.reshape(batch_size, -1) @ x.reshape(batch_size, -1).T
    )
    empirical_ntks = jax.tree_util.tree_map(batch_product_fn, jacs)
    empirical_ntk = sum(jax.tree_util.tree_leaves(empirical_ntks))
    entk_diagonal = jnp.diag(empirical_ntk)
    entk_off_diagonal = empirical_ntk - jnp.diag(entk_diagonal)
    entk_spectrum = jnp.linalg.svd(empirical_ntk)[1]
    # Compile statistics
    network_statistics = {'feature_norm': feature_norm}
    network_statistics['param_norm'] = jnp.sum(
        jax.flatten_util.ravel_pytree(online_params)[0] ** 2
    )
    network_statistics['entk_diagonal_mean'] = jnp.mean(entk_diagonal)
    network_statistics['entk_diagonal_std'] = jnp.std(entk_diagonal)
    network_statistics['entk_negative_frac'] = jnp.mean(empirical_ntk < 0)
    network_statistics['entk_positive_frac'] = jnp.mean(empirical_ntk > 0)
    network_statistics['entk_off_diagonal_mean'] = jnp.sum(
        entk_off_diagonal
    ) / (batch_size * (batch_size - 1))
    for t in [0.0, 0.001, 0.01]:
      network_statistics[f'srank_{t}'] = jnp.sum(
          singular_vals / jnp.max(singular_vals) > t
      )
      network_statistics[f'dormant_{t}'] = jnp.sum(feature_vars <= t)
      network_statistics[f'entk_srank_{t}'] = jnp.sum(
          entk_spectrum / jnp.max(entk_spectrum) > t
      )
    td_losses, _ = per_sample_loss_fn(online_params, target)
    network_statistics['td_error_var'] = jnp.var(td_losses)
    network_statistics['td_error_mean'] = jnp.mean(td_losses)
    q_values = jax.vmap(q_online, in_axes=(0, None))(
        states, jax.random.PRNGKey(0)
    ).q_values
    network_statistics['q_mean'] = jnp.mean(q_values)
    network_statistics['q_std'] = jnp.std(q_values)
    train_returns.update({'network_statistics': network_statistics})
  return train_returns


@gin.configurable
class JaxFullRainbowMoEAgent(full_rainbow_agent.JaxFullRainbowAgent):
  """A compact implementation of agents for Atari 100k with MoEs."""

  def __init__(
      self,
      num_actions,
      summary_writer=None,
      seed=None,
      log_moe_statistics=False,
      log_network_statistics=False,
  ):
    """Creates the Rainbow-based agent for Atari 100k benchmark with MoEs."""
    super().__init__(
        num_actions=num_actions, seed=seed, summary_writer=summary_writer
    )
    self.log_moe_statistics = log_moe_statistics
    self.log_network_statistics = log_network_statistics
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
    states = self.preprocess_fn(self.replay_elements['state'])
    next_states = self.preprocess_fn(self.replay_elements['next_state'])

    if self._replay_scheme == 'prioritized':
      probs = self.replay_elements['sampling_probabilities']
      # Weight the loss by the inverse priorities.
      loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
      loss_weights /= jnp.max(loss_weights)
    else:
      # Uniform weights if not using prioritized replay.
      loss_weights = jnp.ones(states.shape[0])

    should_log_network_statistics = (
        self.log_network_statistics
        and (self.training_steps % self.summary_writing_frequency) == 0
    )
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
        should_log_network_statistics,
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
        if self.log_network_statistics:
          for k, v in train_returns['network_statistics'].items():
            statistics.append(
                statistics_instance.StatisticsInstance(
                    k, np.asarray(v), type='scalar', step=self.training_steps
                )
            )
        self.collector_dispatcher.write(
            statistics, collector_allowlist=self._collector_allowlist
        )

# coding=utf-8
# Copyright 2023 ReDo authors.
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
"""Variant of Atari100k JaxRainbow that supports neurons and layer recycling."""

from dopamine.jax.agents.full_rainbow import full_rainbow_agent
from dopamine.labs.atari_100k import atari_100k_rainbow_agent
from dopamine.labs.redo import networks
from dopamine.labs.redo import weight_recyclers
from dopamine.metrics import statistics_instance
import gin
import jax
import jax.numpy as jnp
import optax


@gin.configurable
class RecycledAtari100kRainbowAgent(
    atari_100k_rainbow_agent.Atari100kRainbowAgent
):
  """Atari100k Ranibow Agent with recycled parameters."""

  def __init__(
      self,
      num_actions,
      network='nature',
      reset_mode=None,
      width=1,
      batch_size_statistics=256,
      target_update_strategy='training_step',
      weight_decay=0.0,
      summary_writer=None,
      is_debugging=False,
  ):
    network_name = network
    if network == 'nature':
      network = networks.FullRainbowNetwork
    else:
      raise ValueError(f'Invalid network: {network}')
    super().__init__(
        num_actions, network=network, summary_writer=summary_writer
    )

    # For DrQ or DER agents dueling is enabled. In this case, the last
    # dense layer is connected to two layers. We put these two layers in one
    # list to represent that they are not consecutive layers.
    if network.dueling:
      new_layer_names = network.layer_names[:-2]
      new_layer_names.append([network.layer_names[-2], network.layer_names[-1]])
      network.layer_names = new_layer_names

    if weight_decay > 0:
      # TODO(gsokar) we may compare the performance with adamw.
      # lets keep it under a condition till we check its effect.
      self.optimizer = optax.chain(
          optax.add_decayed_weights(weight_decay), self.optimizer
      )
      self.optimizer_state = self.optimizer.init(self.online_params)

    self.batch_size_statistics = batch_size_statistics
    self.target_update_strategy = target_update_strategy
    self.is_debugging = is_debugging
    if reset_mode is not None:
      if reset_mode == 'neurons':
        self.weight_recycler = weight_recyclers.NeuronRecycler(
            network.layer_names, network=network_name
        )
      elif reset_mode == 'weights':
        self.weight_recycler = weight_recyclers.LayerReset(network.layer_names)
      else:
        raise ValueError(f'Invalid reset mode: {reset_mode}')
    else:
      self.weight_recycler = weight_recyclers.BaseRecycler(network.layer_names)

  def _log_stats(self, log_dict, step):
    if log_dict is None:
      return
    stats = []
    for k, v in log_dict.items():
      stats.append(statistics_instance.StatisticsInstance(k, v, step=step))
    self.collector_dispatcher.write(
        stats, collector_allowlist=self._collector_allowlist
    )

  def _train_step(self):
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        for step in range(self._num_updates_per_train_step):
          self.gradient_step = (
              (self.training_steps // self.update_period)
              * self._num_updates_per_train_step
          ) + step
          self._training_step_update()
          if self.target_update_strategy == 'update_step':
            if self.gradient_step % self.target_update_period == 0:
              self._sync_weights()

      # The original agent updates target based on training steps. We need to
      # analyze whether we need to change it to gradient_step in case of
      # high replay ratio (i.e., _num_updates_per_train_step > 1) and recycling.
      if self.target_update_strategy == 'training_step':
        if self.training_steps % self.target_update_period == 0:
          self._sync_weights()

    self.training_steps += 1

  def _training_step_update(self):
    """Gradient update."""
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

    (self.optimizer_state, self.online_params, loss, mean_loss, self._rng) = (
        full_rainbow_agent.train(
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
            self._rng,
        )
    )

    if self._replay_scheme == 'prioritized':
      self._replay.set_priority(
          self.replay_elements['indices'], jnp.sqrt(loss + 1e-10)
      )

    # We are using # gradient_update_steps in our calculations and logging.
    update_step = self.gradient_step
    is_logging = (
        update_step > 0 and update_step % self.summary_writing_frequency == 0
    )
    if is_logging:
      self._log_stats({'CrossEntropyLoss': float(mean_loss)}, update_step)

    self.online_params = self._weight_recycle(update_step, self.online_params)

  def _weight_recycle(self, update_step, online_params):
    # Neuron/layer recycling starts if reset_mode is not None.
    # Otherwise, we log dead neurons over training for standard agent.
    is_intermediated = self.weight_recycler.is_intermediated_required(
        update_step
    )
    # get intermediate activation per layer to calculate neuron score
    intermediates = (
        self.get_intermediates(online_params) if is_intermediated else None
    )
    log_dict_neurons = self.weight_recycler.maybe_log_deadneurons(
        update_step, intermediates
    )
    # logging dead neurons.
    self._log_stats(log_dict_neurons, update_step)
    if self.is_debugging:
      log_dict_intersected = (
          self.weight_recycler.intersected_dead_neurons_with_last_reset(
              intermediates, update_step
          )
      )
      self._log_stats(log_dict_intersected, update_step)

    # Neuron/layer recyling.
    self._rng, key = jax.random.split(self._rng)
    online_params, opt_state = self.weight_recycler.maybe_update_weights(
        update_step, intermediates, online_params, key, self.optimizer_state
    )
    self.optimizer_state = opt_state
    return online_params

  def _sample_batch_for_statistics(self):
    samples = self._replay.sample_transition_batch(
        batch_size=self.batch_size_statistics
    )
    types = self._replay.get_transition_elements()
    for element, element_type in zip(samples, types):
      if element_type.name == 'state':
        states = self.preprocess_fn(element)
        break
    return states

  def get_intermediates(self, online_params):
    # TODO(gsokar) add a check if batch_size equals batch_size_statistics
    # then no need to sample a new batch from buffer.
    batch = self._sample_batch_for_statistics()

    def apply_data(x):
      filter_rep = lambda l, _: l.name is not None and 'act' in l.name
      self._rng, key = jax.random.split(self._rng)
      return self.network_def.apply(
          online_params,
          x,
          support=self._support,
          key=key,
          capture_intermediates=filter_rep,
          mutable=['intermediates'],
      )

    _, state = jax.vmap(apply_data)(batch)
    return state['intermediates']

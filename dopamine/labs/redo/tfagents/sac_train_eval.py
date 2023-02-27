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
r"""Train and Eval SAC.
"""

import functools
import os

from absl import app
from absl import flags
from absl import logging
import gin
import numpy as np
import reverb
from rigl.rl.tfagents import sparse_tanh_normal_projection_network
import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.agents.sac import sac_agent
from tf_agents.environments import suite_mujoco
from tf_agents.keras_layers import inner_reshape
from tf_agents.metrics import py_metrics
from tf_agents.networks import nest_map
from tf_agents.networks import sequential
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.utils import common
from tf_agents.utils import object_identity

FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer(
    'reverb_port', None,
    'Port for reverb server, if None, use a randomly chosen unused port.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', [], 'Gin binding parameters.')

# Env params
flags.DEFINE_bool('is_atari', False, 'Whether the env is an atari game.')
flags.DEFINE_bool('is_mujoco', False, 'Whether the env is a mujoco game.')
flags.DEFINE_bool('is_classic', False,
                  'Whether the env is a classic control game.')
flags.DEFINE_float(
    'average_last_fraction', 0.1,
    'Tells what fraction latest evaluation scores are averaged. This is used'
    ' to reduce variance.')

original_call = tf.keras.layers.Dense.call


def custom_call(self, inputs):
  output = original_call(self, inputs)
  self.last_activation = output
  return output


def get_all_layers(model, filter_fn=lambda _: True):
  """Gets all layers of a model and layers of a layer if it is a keras.Model."""
  all_layers = []
  for l in model.layers:
    if hasattr(l, 'layers'):
      all_layers.extend(get_all_layers(l, filter_fn=filter_fn))
    elif filter_fn(l):
      all_layers.append(l)
  return all_layers


# This is a hacky way of changing the behaviour of a class function.
MyDenseLayer = tf.keras.layers.Dense
MyDenseLayer.call = custom_call

dense = functools.partial(
    MyDenseLayer,
    activation=tf.keras.activations.relu,
    kernel_initializer='glorot_uniform')


def is_dense_layer(layer):
  return isinstance(layer, MyDenseLayer)


def scale_width(num_units: int, width: float):
  assert width > 0
  return int(max(1, num_units * width))


def get_intermedieates(*networks):
  """Retrieves activations of the Dense layers in the network."""
  result = {}

  def _helper(network):
    if isinstance(network, MyDenseLayer):
      result[network.name] = network.last_activation
    elif hasattr(network, 'layers'):
      for layer in network.layers:
        _helper(layer)

  for network in networks:
    _helper(network)
  return result


def create_fc_layers(layer_units, width=1.0, weight_decay=0):
  layers = [
      dense(
          scale_width(num_units, width=width),
          kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
      )
      for num_units in layer_units
  ]
  return layers


def create_identity_layer():
  return tf.keras.layers.Lambda(lambda x: x)


def create_sequential_critic_network(
    obs_fc_layer_units,
    action_fc_layer_units,
    joint_fc_layer_units,
    width: float = 1.0,
    weight_decay: float = 0.0,
):
  """Create a sequential critic network."""
  # Split the inputs into observations and actions.
  def split_inputs(inputs):
    return {'observation': inputs[0], 'action': inputs[1]}

  # Create an observation network layers.
  obs_network_layers = (
      create_fc_layers(obs_fc_layer_units, width=width,
                       weight_decay=weight_decay)
      if obs_fc_layer_units else None)

  # Create an action network layers.
  action_network_layers = (
      create_fc_layers(action_fc_layer_units, width=width,
                       weight_decay=weight_decay)
      if action_fc_layer_units else None)

  # Create a joint network layers.
  joint_network_layers = (
      create_fc_layers(joint_fc_layer_units, width=width,
                       weight_decay=weight_decay)
      if joint_fc_layer_units else None)

  # Final layer.
  value_layer = MyDenseLayer(
      1,
      kernel_initializer='glorot_uniform',
      kernel_regularizer=tf.keras.regularizers.L2(weight_decay))

  layer_list = [obs_network_layers, action_network_layers,
                joint_network_layers]

  # Convert layer_list to sequential or identity lambdas:
  module_list = [create_identity_layer() if layers is None else
                 sequential.Sequential(layers)
                 for layers in layer_list]
  obs_network, action_network, joint_network = module_list

  return sequential.Sequential([
      tf.keras.layers.Lambda(split_inputs),
      nest_map.NestMap({
          'observation': obs_network,
          'action': action_network
      }),
      nest_map.NestFlatten(),
      tf.keras.layers.Concatenate(),
      joint_network,
      value_layer,
      inner_reshape.InnerReshape(current_shape=[1], new_shape=[])
  ], name='sequential_critic')


class _TanhNormalProjectionNetworkWrapper(
    sparse_tanh_normal_projection_network.SparseTanhNormalProjectionNetwork):
  """Wrapper to pass predefined `outer_rank` to underlying projection net."""

  def __init__(self, sample_spec, predefined_outer_rank=1, weight_decay=0.0):
    super(_TanhNormalProjectionNetworkWrapper, self).__init__(
        sample_spec=sample_spec,
        weight_decay=weight_decay)
    self.predefined_outer_rank = predefined_outer_rank

  def call(self, inputs, network_state=(), **kwargs):
    kwargs['outer_rank'] = self.predefined_outer_rank
    if 'step_type' in kwargs:
      del kwargs['step_type']
    return super(_TanhNormalProjectionNetworkWrapper,
                 self).call(inputs, **kwargs)


def create_sequential_actor_network(
    actor_fc_layers,
    action_tensor_spec,
    width: float = 1.0,
    weight_decay: float = 0.0,
):
  """Create a sequential actor network."""
  def tile_as_nest(non_nested_output):
    return tf.nest.map_structure(lambda _: non_nested_output,
                                 action_tensor_spec)

  dense_layers = [
      dense(
          scale_width(num_units, width=width),
          kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
      )
      for num_units in actor_fc_layers
  ]
  tanh_normal_projection_network_fn = functools.partial(
      _TanhNormalProjectionNetworkWrapper,
      weight_decay=weight_decay)
  last_layer = nest_map.NestMap(
      tf.nest.map_structure(tanh_normal_projection_network_fn,
                            action_tensor_spec))

  return sequential.Sequential(
      dense_layers +
      [tf.keras.layers.Lambda(tile_as_nest)] + [last_layer])


@gin.configurable
class RecycledSacAgent(sac_agent.SacAgent):
  """Wrapped DqnAgent that supports recycled training."""

  def __init__(
      self,
      time_step_spec,
      action_spec,
      *args,
      reset_mode=None,
      reset_freq=200000,
      reset_frac=0.0,
      reset_target_models=False,
      reset_actor=True,
      reset_critic=True,
      reset_algo='low_score',
      neuron_score_algo='activation',
      dead_neuron_threshold=0.0,
      scale_recycled_weights=False,
      recycled_incoming_scaler=1,
      recycled_outgoing_scaler=0,
      init_method='zero',
      log_interval=10000,
      log_batch_size=256,
      reset_freq_scale=1,
      **kwargs,
  ):
    super().__init__(time_step_spec,
                     action_spec,
                     *args,
                     **kwargs)
    self.reset_mode = reset_mode
    self.reset_freq = int(reset_freq * reset_freq_scale)
    self.reset_frac = reset_frac
    self.reset_algo = reset_algo
    self.reset_target_models = reset_target_models
    self.reset_actor = reset_actor
    self.reset_critic = reset_critic
    self.neuron_score_algo = neuron_score_algo
    self.dead_neuron_threshold = dead_neuron_threshold
    self.scale_recycled_weights = scale_recycled_weights
    self.recycled_incoming_scaler = recycled_incoming_scaler
    self.recycled_outgoing_scaler = recycled_outgoing_scaler
    self.init_method = init_method
    self.log_interval = log_interval
    self.log_batch_size = log_batch_size

    net_observation_spec = time_step_spec.observation
    critic_spec = (net_observation_spec, action_spec)
    self._target_critic_network_1 = (
        common.maybe_copy_target_network_with_checks(
            self._critic_network_1,
            None,
            input_spec=critic_spec,
            name='TargetCriticNetwork1'))
    self._target_critic_network_1 = (
        common.maybe_copy_target_network_with_checks(
            self._critic_network_2,
            None,
            input_spec=critic_spec,
            name='TargetCriticNetwork2'))

  def _train(self, experience, weights):
    """Returns a train op to update the agent's networks.

    This method trains with the provided batched experience.

    Args:
      experience: A time-stacked trajectory object.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.

    Returns:
      A train_op.

    Raises:
      ValueError: If optimizers are None and no default value was provided to
        the constructor.
    """
    tf.summary.experimental.set_step(self.train_step_counter)
    transition = self._as_transition(experience)
    time_steps, policy_steps, next_time_steps = transition
    actions = policy_steps.action

    trainable_critic_variables = list(
        object_identity.ObjectIdentitySet(
            self._critic_network_1.trainable_variables +
            self._critic_network_2.trainable_variables))

    with tf.GradientTape(
        watch_accessed_variables=False, persistent=True) as tape:
      assert trainable_critic_variables, ('No trainable critic variables to '
                                          'optimize.')
      tape.watch(trainable_critic_variables)
      critic_loss = self._critic_loss_weight * self.critic_loss(
          time_steps,
          actions,
          next_time_steps,
          td_errors_loss_fn=self._td_errors_loss_fn,
          gamma=self._gamma,
          reward_scale_factor=self._reward_scale_factor,
          weights=weights,
          training=True)

      self.critic1_act = get_intermedieates(self._critic_network_1)
      self.critic2_act = get_intermedieates(self._critic_network_2)
      self.critic1_act_grads = (
          tape.gradient(critic_loss, self.critic1_act)
          if self.neuron_score_algo == 'activation_grad' else None)
      self.critic2_act_grads = (
          tape.gradient(critic_loss, self.critic2_act)
          if self.neuron_score_algo == 'activation_grad' else None)

    tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
    critic_grads = tape.gradient(critic_loss, trainable_critic_variables)
    self._apply_gradients(critic_grads, trainable_critic_variables,
                          self._critic_optimizer)

    trainable_actor_variables = self._actor_network.trainable_variables
    with tf.GradientTape(
        watch_accessed_variables=False, persistent=True) as tape:
      assert trainable_actor_variables, ('No trainable actor variables to '
                                         'optimize.')
      tape.watch(trainable_actor_variables)
      actor_loss = self._actor_loss_weight*self.actor_loss(
          time_steps, weights=weights, training=True)

      self.actor_act = get_intermedieates(self._actor_network)
      self.actor_act_grads = (
          tape.gradient(actor_loss, self.actor_act)
          if self.neuron_score_algo == 'activation_grad' else None)

    tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
    actor_grads = tape.gradient(actor_loss, trainable_actor_variables)
    self._apply_gradients(actor_grads, trainable_actor_variables,
                          self._actor_optimizer)

    is_deadneurons_log_step = self.is_dead_neurons_log_iter()
    tf.cond(is_deadneurons_log_step, self.log_deadneurons_models, lambda: None)

    if self.reset_mode:
      is_reset_step = self.is_reset_iter()
      tf.cond(is_reset_step, self.reset_models, lambda: None)

    is_logging = self.train_step_counter % self.log_interval == 0
    log_w_mean_c1 = functools.partial(self.log_weights_mean, 'critic_1',
                                      self._critic_network_1)
    tf.cond(is_logging, log_w_mean_c1, lambda: None)
    log_w_mean_c2 = functools.partial(self.log_weights_mean, 'critic_2',
                                      self._critic_network_2)
    tf.cond(is_logging, log_w_mean_c2, lambda: None)
    log_w_mean_actor = functools.partial(self.log_weights_mean, 'actor',
                                         self._actor_network)
    tf.cond(is_logging, log_w_mean_actor, lambda: None)

    alpha_variable = [self._log_alpha]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert alpha_variable, 'No alpha variable to optimize.'
      tape.watch(alpha_variable)
      alpha_loss = self._alpha_loss_weight * self.alpha_loss(
          time_steps, weights=weights, training=True)
    tf.debugging.check_numerics(alpha_loss, 'Alpha loss is inf or nan.')
    alpha_grads = tape.gradient(alpha_loss, alpha_variable)
    self._apply_gradients(alpha_grads, alpha_variable, self._alpha_optimizer)

    with tf.name_scope('Losses'):
      tf.compat.v2.summary.scalar(
          name='critic_loss', data=critic_loss, step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='actor_loss', data=actor_loss, step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='alpha_loss', data=alpha_loss, step=self.train_step_counter)

    self.train_step_counter.assign_add(1)
    self._update_target()

    total_loss = critic_loss + actor_loss + alpha_loss

    extra = sac_agent.SacLossInfo(
        critic_loss=critic_loss, actor_loss=actor_loss, alpha_loss=alpha_loss)

    return tf_agent.LossInfo(loss=total_loss, extra=extra)

  # for debugging
  def log_weights_mean(self, model_name, model):
    model_layers = get_all_layers(model)
    for layer in model_layers:
      for var in layer.weights:
        mean_w_mag = tf.reduce_mean(tf.math.abs(var))
        with tf.name_scope('w_mean/' + model_name + '/'):
          tf.compat.v2.summary.scalar(layer.name, mean_w_mag)

  def is_dead_neurons_log_iter(self):
    is_iter = tf.logical_and(self.train_step_counter > 0,
                             self.train_step_counter % self.log_interval == 0)
    return is_iter

  def log_deadneurons_models(self):
    self.log_dead_neurons_count(self._critic_network_1, 'critic_1',
                                self.critic1_act, self.critic1_act_grads)
    self.log_dead_neurons_count(self._critic_network_2, 'critic_2',
                                self.critic2_act, self.critic2_act_grads)
    self.log_dead_neurons_count(self._actor_network, 'actor', self.actor_act,
                                self.actor_act_grads)

  def calculate_neuron_score_all_layers(self, model, intermediate_act,
                                        intermediate_act_grad):
    all_layers_score = {}
    if self.neuron_score_algo == 'activation':
      for act_key, act_value in intermediate_act.items():
        self.log_batch_size = min(self.log_batch_size, act_value.shape[0])
        neurons_score = tf.reduce_mean(
            tf.math.abs(act_value[:self.log_batch_size, :]), axis=0)
        neurons_score = neurons_score / (tf.reduce_mean(neurons_score) + 1e-9)
        all_layers_score[act_key] = neurons_score
    elif self.neuron_score_algo == 'activation_grad':
      for act, act_grad in zip(intermediate_act.items(),
                               intermediate_act_grad.items()):
        act_k, act_v = act
        _, act_grad_v = act_grad
        self.log_batch_size = min(self.log_batch_size, act_v.shape[0])
        act_multi_grad = tf.math.abs(
            tf.multiply(act_v[:self.log_batch_size, :],
                        act_grad_v[:self.log_batch_size, :]))
        neurons_score = tf.reduce_mean(act_multi_grad, axis=0)
        neurons_score = neurons_score / (tf.reduce_mean(neurons_score) + 1e-9)
        all_layers_score[act_k] = neurons_score
    elif self.neuron_score_algo == 'connected_weights':
      model_layers = get_all_layers(model, filter_fn=is_dense_layer)
      for layer, next_layer in zip(model_layers[:-1], model_layers[1:]):
        incoming_var = layer.kernel
        outgoing_var = next_layer.kernel
        # incoming_var is KxN, outgoing_var is NxM.
        # neurons_score is N dimensional
        neurons_score = tf.reduce_sum(tf.math.abs(incoming_var), axis=0)
        neurons_score *= tf.reduce_sum(tf.math.abs(outgoing_var), axis=1)
        neurons_score = neurons_score / (tf.reduce_mean(neurons_score) + 1e-9)
        all_layers_score[layer.name] = neurons_score
    else:
      raise ValueError('neuron_score_algo:%s  is not valid.' %
                       self.neuron_score_algo)
    return all_layers_score

  def log_dead_neurons_count(self, model, network_name, intermediate_act,
                             intermediate_act_grad):
    """log the number of dead neurons per hidden layer in a network.

    Args:
      model: network
      network_name: name of the network
      intermediate_act: dict of layers activation
      intermediate_act_grad: dict of layers grad of activation
    """
    total_dead_neurons = 0
    total_hidden_count = 0
    all_layers_score = self.calculate_neuron_score_all_layers(
        model, intermediate_act, intermediate_act_grad)

    for layer_count, (layer_name,
                      layer_score) in enumerate(all_layers_score.items()):
      _, num_dead_neurons = self.get_dead_neurons(layer_score)
      total_dead_neurons += num_dead_neurons
      total_hidden_count += layer_score.shape[0]
      with tf.name_scope('dead_neurons/' + network_name + '/'):
        tf.compat.v2.summary.scalar(
            'layer_' + str(layer_count + 1) + '_' + layer_name,
            num_dead_neurons)

      log_act_hist = functools.partial(
          self.log_histogram, layer_name + str(self.train_step_counter),
          layer_score)
      tf.cond(self.train_step_counter % 100000 == 0, log_act_hist, lambda: None)

    with tf.name_scope('dead_neurons/' + network_name + '/'):
      tf.compat.v2.summary.scalar('total_dead_neurons', total_dead_neurons)
    with tf.name_scope('dead_neurons/' + network_name + '/'):
      tf.compat.v2.summary.scalar(
          'dead_neurons_percentage',
          (total_dead_neurons / total_hidden_count) * 100,
      )

  def log_histogram(self, name, activation):
    tf.summary.histogram(name, activation, step=self.train_step_counter)

  def is_reset_iter(self):
    """Returns true if it is a valid reset step."""
    return tf.logical_and(self.train_step_counter > 0,
                          self.train_step_counter % self.reset_freq == 0)

  def reset_models(self):
    if self.reset_mode == 'weights':
      self.reset_model_weights()
    elif self.reset_mode == 'neurons':
      self.reset_model_neurons()
    else:
      raise ValueError('reset_mode:%s  is not valid.' % self.reset_mode)

  def reset_momentum(self, optimizer, var, mask):
    for s_name in optimizer.get_slot_names():
      # Momentum variable for example, we reset the aggregated values to zero.
      optim_var = optimizer.get_slot(var, s_name)
      new_values = tf.where(mask == 1, tf.zeros_like(optim_var), optim_var)
      optim_var.assign(new_values)

  def reset_model_weights(self):
    self.reset_weights(self._critic_optimizer, self._critic_network_1,
                       self._target_critic_network_1)
    self.reset_weights(self._critic_optimizer, self._critic_network_2,
                       self._target_critic_network_2)
    self.reset_weights(self._actor_optimizer, self._actor_network)

  def reset_weights(self, optimizer, model, target_model=None):
    logging.info('weights reset')
    model_layers = get_all_layers(model)
    target_model_layers = (
        model_layers if target_model is None else get_all_layers(target_model)
    )
    assert (len(model_layers) == len(target_model_layers))
    for layer, target_layer in zip(model_layers, target_model_layers):
      assert (len(layer.weights) == len(target_layer.weights))
      for var, target_var in zip(layer.weights, target_layer.weights):
        abs_weights = tf.math.abs(var)
        k = tf.dtypes.cast(
            tf.math.maximum(
                tf.math.round(
                    tf.dtypes.cast(tf.size(abs_weights), tf.float32) *
                    self.reset_frac), 1), tf.int32)
        w_initializer = layer.kernel_initializer
        logging.info('shape %s', var.shape)
        logging.info('w_initializer%s', w_initializer)

        b_initializer = layer.bias_initializer
        new_weights, mask = self.create_new_weights(k, var, abs_weights,
                                                    w_initializer,
                                                    b_initializer)
        var.assign(new_weights)
        if target_model is not None:
          new_target_weights = tf.where(mask == 1, new_weights, target_var)
          target_var.assign(new_target_weights)

        # reset momentum for the weights that are resetted
        if len(var.get_shape().as_list()) > 1:
          self.reset_momentum(optimizer, var, mask)

  def create_new_weights(self, k, weights, abs_weights, w_initializer,
                         b_initializer):
    w_shape = weights.get_shape().as_list()
    if len(w_shape) == 2:
      random_weights = w_initializer(weights.shape)
      if self.reset_algo == 'random':
        score = tf.random.uniform(weights.shape, 0, 1)
      elif self.reset_algo == 'low_score':
        score = -1 * abs_weights
      elif self.reset_algo == 'high_score':
        score = abs_weights
      else:
        raise ValueError('reset_alg:%s  is not valid.' % self.reset_algo)
      values, _ = tf.math.top_k(tf.reshape(score, [-1]), k=tf.size(score))
      threshold_value = tf.gather(values, k - 1)
      mask = tf.where(
          tf.math.greater_equal(score, threshold_value),
          tf.ones_like(score, dtype=tf.int32),
          tf.zeros_like(score, dtype=tf.int32))
      new_weights = tf.where(mask == 1, random_weights, weights)

    # bias
    elif len(w_shape) == 1:
      if self.reset_frac == 1.0:
        random_weights = b_initializer(weights.shape)
        mask = tf.ones_like(weights)
      # keep the bias if a subset of the weights is updated
      else:
        random_weights = weights
        mask = tf.zeros_like(weights)
      new_weights = random_weights
    return new_weights, mask

  def reset_model_neurons(self):
    if self.reset_critic:
      tf.print('critic reset')
      self.reset_dead_neurons(self._critic_optimizer, self.critic1_act,
                              self.critic1_act_grads, self._critic_network_1,
                              self._target_critic_network_1)
      self.reset_dead_neurons(self._critic_optimizer, self.critic2_act,
                              self.critic2_act_grads, self._critic_network_2,
                              self._target_critic_network_2)
    if self.reset_actor:
      tf.print('actor reset')
      self.reset_dead_neurons(self._actor_optimizer, self.actor_act,
                              self.actor_act_grads, self._actor_network)

  def get_dead_neurons(self, neuron_score):
    mask = tf.where(
        neuron_score <= self.dead_neuron_threshold,
        tf.ones_like(neuron_score, dtype=tf.int32),
        tf.zeros_like(neuron_score, dtype=tf.int32),
    )
    num_dead_neurons = tf.reduce_sum(mask)
    return mask, num_dead_neurons

  def get_mask_dead_neurons_weights(self, act, act_grad, model):
    model_layers = get_all_layers(model, filter_fn=is_dense_layer)
    all_layers_score = self.calculate_neuron_score_all_layers(
        model, act, act_grad)
    incoming_masks = {}
    outgoing_masks = {}
    dead_neuron_masks = {}
    for layer, next_layer, in zip(model_layers[:-1], model_layers[1:]):
      incoming_var = layer.kernel
      outgoing_var = next_layer.kernel
      neurons_score = all_layers_score[layer.name]
      # TODO(gsokar) implement random and high score based on threshold.
      if self.reset_algo == 'low_score':
        score = neurons_score
      else:
        raise ValueError('reset_alg:%s  is not valid.' % self.reset_algo)
      mask, _ = self.get_dead_neurons(score)
      incoming_mask, outgoing_mask = self.create_mask_helper(
          mask, incoming_var.shape[0], outgoing_var.shape[1])
      incoming_masks[layer.name] = incoming_mask
      outgoing_masks[next_layer.name] = outgoing_mask
      dead_neuron_masks[layer.name] = mask
    return dead_neuron_masks, incoming_masks, outgoing_masks

  def reset_dead_neurons(self,
                         optimizer,
                         act,
                         act_grad,
                         model,
                         target_model=None):
    """Recycle the dead neurons by reinitializing its weights.

    Args:
      optimizer: optimizer of a network weights
      act: activation of each layer that is used to calculate the neuron score
        for some score metrics
      act_grad: gradient of the activation of each hidden layer
      model: behavior network
      target_model: target network. When there is no target model (e.g. actor
        network case), we iterate only on the behavior network layers
    """
    model_layers = get_all_layers(model, filter_fn=is_dense_layer)
    target_model_layers = (
        model_layers
        if target_model is None
        else get_all_layers(target_model, filter_fn=is_dense_layer)
    )
    dead_neuron_masks, incoming_masks, outgoing_masks = (
        self.get_mask_dead_neurons_weights(act, act_grad, model))
    # update incoming weights
    for layer, target_layer in zip(model_layers[:-1], target_model_layers[:-1]):
      incoming_mask = incoming_masks[layer.name]
      incoming_var = layer.kernel
      incoming_target_var = target_layer.kernel
      w_initializer = type(layer.kernel_initializer)(
          seed=self.train_step_counter
      )
      new_in_weights = w_initializer(incoming_var.shape)

      if self.scale_recycled_weights:
        neuron_mask = dead_neuron_masks[layer.name]
        new_in_weights = self._rescale_weights(neuron_mask, incoming_var,
                                               new_in_weights, 0,
                                               self.recycled_incoming_scaler)

      new_in_weights = tf.where(incoming_mask == 1, new_in_weights,
                                incoming_var)

      incoming_var.assign(new_in_weights)
      if self.reset_target_models and target_model is not None:
        new_target_in_weights = tf.where(incoming_mask == 1, new_in_weights,
                                         incoming_target_var)
        incoming_target_var.assign(new_target_in_weights)
      # reset momentum for the weights that are resetted
      self.reset_momentum(optimizer, incoming_var, incoming_mask)

      # reset bias of dead neurons
      new_bias = tf.zeros_like(layer.bias)
      new_bias = tf.where(
          tf.math.equal(dead_neuron_masks[layer.name], 1), new_bias, layer.bias)
      layer.bias.assign(new_bias)

    # update outgoing weights
    # model layers returns list of each layer in the model.
    # e.g. model_layers = [dense_1, dense_2, output layer]
    # updating the outgoing connections of the first hidden layer (dense_1)
    # means that we update the weights of the second hidden layer (dense_2).
    # in case of weight scaling, we identify the non dead neurons in dense_1
    # to get the average norm that will be used to scale the outgoing weights
    # of dense_1 (weights of dense_2).
    # model[1:] = [dense_2, output layer] to access weights
    # model[:-1] = [dense_1, dense_2] to access the corresponding dead neurons.
    for layer, next_layer, target_next_layer in zip(model_layers[:-1],
                                                    model_layers[1:],
                                                    target_model_layers[1:]):
      outgoing_mask = outgoing_masks[next_layer.name]
      outgoing_var = next_layer.kernel
      outgoing_target_var = target_next_layer.kernel
      if self.init_method == 'random':
        w_initializer = type(next_layer.kernel_initializer)(
            seed=self.train_step_counter
        )
        new_out_weights = w_initializer(outgoing_var.shape)
        # if we are interested in scaling the ingoing only,
        # keep recycled_outgoing_scaler with the default 0.
        if self.scale_recycled_weights and self.recycled_outgoing_scaler > 0:
          neuron_mask = dead_neuron_masks[layer.name]
          new_out_weights = self._rescale_weights(neuron_mask, outgoing_var,
                                                  new_out_weights, 1,
                                                  self.recycled_outgoing_scaler)

      elif self.init_method == 'zero':
        new_out_weights = tf.zeros_like(outgoing_var)
      else:
        raise ValueError('init_method:%s  is not valid.' % self.init_method)

      new_out_weights = tf.where(outgoing_mask == 1, new_out_weights,
                                 outgoing_var)
      outgoing_var.assign(new_out_weights)
      if self.reset_target_models and target_model is not None:
        new_target_out_weights = tf.where(outgoing_mask == 1, new_out_weights,
                                          outgoing_target_var)
        outgoing_target_var.assign(new_target_out_weights)
      self.reset_momentum(optimizer, outgoing_var, outgoing_mask)

  def _rescale_weights(self, neuron_mask, var, new_weights, axis, scaler):
    non_dead_count = neuron_mask.shape[0] - tf.reduce_sum(neuron_mask)
    non_recycled_norm = tf.reduce_sum((tf.norm(var, axis=axis) * tf.cast(
        1 - neuron_mask, tf.float32))) / tf.cast(non_dead_count, tf.float32)

    new_recycled_norm = non_recycled_norm * scaler
    weights_norm_per_neuron = tf.norm(new_weights, axis=axis, keepdims=True)
    new_weights = new_weights / weights_norm_per_neuron
    new_weights = new_recycled_norm * new_weights
    return new_weights

  def create_mask_helper(self, neuron_mask, prev_layer_size, next_layer_size):
    """create masks of the weights that will be restarted.

    take the mask of neurons and create the masked ingoing,
    and outgoing connections based on neuron mask.

    Args:
      neuron_mask: mask of a neurons in one layer i
      prev_layer_size: size of layer i-1
      next_layer_size: size of layer i+1

    Returns:
      mask of ingoing and outgoing weights
    """
    neuron_mask = tf.expand_dims(neuron_mask, 1)
    # mask of ingoing weights
    tile_multiple = tf.constant([1, prev_layer_size], tf.int32)
    ingoing_w_mask_t = tf.tile(neuron_mask, tile_multiple)
    ingoing_w_mask = tf.transpose(ingoing_w_mask_t)
    # mask of outgoing weights
    tile_multiple = tf.constant([1, next_layer_size], tf.int32)
    outgoing_w_mask = tf.tile(neuron_mask, tile_multiple)
    return ingoing_w_mask, outgoing_w_mask


@gin.configurable
def train_eval(
    root_dir,
    strategy: tf.distribute.Strategy,
    env_name='HalfCheetah-v2',
    # Training params
    initial_collect_steps=10000,
    heavy_priming: bool = False,
    heavy_priming_num_iterations=100000,
    weights_update_per_interaction=1,
    scale_lr_using_rr=False,
    scale_reset_freq=False,
    num_iterations=1000000,
    actor_fc_layers=(256, 256),
    critic_obs_fc_layers=None,
    critic_action_fc_layers=None,
    critic_joint_fc_layers=(256, 256),
    # Agent params
    batch_size=256,
    learning_rate_overwrite=None,
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    alpha_learning_rate=3e-4,
    gamma=0.99,
    target_update_tau=0.005,
    target_update_period=1,
    reward_scale_factor=0.1,
    # Replay params
    reverb_port=None,
    replay_capacity=1000000,
    # Others
    policy_save_interval=1000000,
    replay_buffer_save_interval=100000,
    eval_interval=10000,
    eval_episodes=30,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    width: float = 1.0,
    train_mode_actor: str = 'dense',
    train_mode_value: str = 'dense',
    weight_decay: float = 0.0,
    actor_critic_widths_str: str = '',
):
  """Trains and evaluates SAC."""
  assert FLAGS.is_mujoco
  assert isinstance(weights_update_per_interaction, int)
  assert weights_update_per_interaction >= 1
  if learning_rate_overwrite:
    actor_learning_rate = learning_rate_overwrite
    critic_learning_rate = learning_rate_overwrite
    alpha_learning_rate = learning_rate_overwrite
  if scale_lr_using_rr:
    actor_learning_rate /= weights_update_per_interaction
    critic_learning_rate /= weights_update_per_interaction
    alpha_learning_rate /= weights_update_per_interaction
  # BEGIN_GOOGLE_INTERNAL
  xm_client = xmanager_api.XManagerApi()
  work_unit = xm_client.get_current_work_unit()
  xm_objective_value_train_reward = work_unit.get_measurement_series(
      label='train_reward')
  xm_objective_value_reward = work_unit.get_measurement_series(label='reward')
  # END_GOOGLE_INTERNAL

  if actor_critic_widths_str:
    actor_critic_widths = [float(s) for s in actor_critic_widths_str.split('_')]
    width_actor = actor_critic_widths[0]
    width_value = actor_critic_widths[1]
  else:
    width_actor = width
    width_value = width

  logging.info('Training SAC on: %s', env_name)
  logging.info('SAC params: train mode actor: %s', train_mode_actor)
  logging.info('SAC params: train mode value: %s', train_mode_value)
  logging.info('SAC params: width: %s', width)
  logging.info('SAC params: actor_critic_widths_str: %s',
               actor_critic_widths_str)
  logging.info('SAC params: width_actor: %s', width_actor)
  logging.info('SAC params: width_value: %s', width_value)
  logging.info('SAC params: weight_decay: %s', weight_decay)

  collect_env = suite_mujoco.load(env_name)
  eval_env = suite_mujoco.load(env_name)

  _, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(collect_env))

  actor_net = create_sequential_actor_network(
      actor_fc_layers=actor_fc_layers,
      action_tensor_spec=action_tensor_spec,
      width=width_actor,
      weight_decay=weight_decay,
  )

  critic_net = create_sequential_critic_network(
      obs_fc_layer_units=critic_obs_fc_layers,
      action_fc_layer_units=critic_action_fc_layers,
      joint_fc_layer_units=critic_joint_fc_layers,
      width=width_value,
      weight_decay=weight_decay,
  )

  if scale_reset_freq:
    reset_freq_scale = weights_update_per_interaction
  else:
    reset_freq_scale = 1
  with strategy.scope():
    train_step = train_utils.create_train_step()
    agent = RecycledSacAgent(
        time_step_spec=time_step_tensor_spec,
        action_spec=action_tensor_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.keras.optimizers.Adam(
            learning_rate=actor_learning_rate
        ),
        critic_optimizer=tf.keras.optimizers.Adam(
            learning_rate=critic_learning_rate
        ),
        alpha_optimizer=tf.keras.optimizers.Adam(
            learning_rate=alpha_learning_rate
        ),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=None,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step,
        reset_freq_scale=reset_freq_scale,
    )

    agent.initialize()
    logging.info('agent initialized.')

  table_name = 'uniform_table'
  table = reverb.Table(
      table_name,
      max_size=replay_capacity,
      sampler=reverb.selectors.Uniform(),
      remover=reverb.selectors.Fifo(),
      rate_limiter=reverb.rate_limiters.MinSize(1))

  reverb_checkpoint_dir = os.path.join(root_dir, learner.TRAIN_DIR,
                                       learner.REPLAY_BUFFER_CHECKPOINT_DIR)
  reverb_checkpointer = reverb.platform.checkpointers_lib.DefaultCheckpointer(
      path=reverb_checkpoint_dir)
  reverb_server = reverb.Server([table],
                                port=reverb_port,
                                checkpointer=reverb_checkpointer)
  reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
      agent.collect_data_spec,
      sequence_length=2,
      table_name=table_name,
      local_server=reverb_server)
  rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
      reverb_replay.py_client,
      table_name,
      sequence_length=2,
      stride_length=1)

  def experience_dataset_fn():
    return reverb_replay.as_dataset(
        sample_batch_size=batch_size, num_steps=2).prefetch(50)

  saved_model_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR)
  env_step_metric = py_metrics.EnvironmentSteps()
  learning_triggers = [
      triggers.PolicySavedModelTrigger(
          saved_model_dir,
          agent,
          train_step,
          interval=policy_save_interval,
          metadata_metrics={triggers.ENV_STEP_METADATA_KEY: env_step_metric}),
      triggers.ReverbCheckpointTrigger(
          train_step,
          interval=replay_buffer_save_interval,
          reverb_client=reverb_replay.py_client),
      triggers.StepPerSecondLogTrigger(train_step, interval=1000),
  ]

  agent_learner = learner.Learner(
      root_dir,
      train_step,
      agent,
      experience_dataset_fn,
      triggers=learning_triggers,
      strategy=strategy)

  random_policy = random_py_policy.RandomPyPolicy(
      collect_env.time_step_spec(), collect_env.action_spec())
  initial_collect_actor = actor.Actor(
      collect_env,
      random_policy,
      train_step,
      steps_per_run=initial_collect_steps,
      observers=[rb_observer])
  logging.info('Doing initial collect.')
  initial_collect_actor.run()

  tf_collect_policy = agent.collect_policy
  collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
      tf_collect_policy, use_tf_function=True)

  collect_actor = actor.Actor(
      collect_env,
      collect_policy,
      train_step,
      steps_per_run=1,
      metrics=actor.collect_metrics(10),
      summary_dir=os.path.join(root_dir, learner.TRAIN_DIR),
      observers=[rb_observer, env_step_metric])

  tf_greedy_policy = greedy_policy.GreedyPolicy(agent.policy)
  eval_greedy_policy = py_tf_eager_policy.PyTFEagerPolicy(
      tf_greedy_policy, use_tf_function=True)

  eval_actor = actor.Actor(
      eval_env,
      eval_greedy_policy,
      train_step,
      episodes_per_run=eval_episodes,
      metrics=actor.eval_metrics(eval_episodes),
      summary_dir=os.path.join(root_dir, 'eval'),
  )

  average_returns = []
  if eval_interval:
    logging.info('Evaluating.')
    eval_actor.run_and_log()
    for metric in eval_actor.metrics:
      if isinstance(metric, py_metrics.AverageReturnMetric):
        average_returns.append(metric._buffer.mean())  # pylint: disable=protected-access

  # heavy priming experiments
  if heavy_priming:
    logging.info('Heavy priming.')
    agent_learner.run(iterations=heavy_priming_num_iterations)

  logging.info('Training.')
  for _ in range(num_iterations):
    collect_actor.run()
    agent_learner.run(iterations=weights_update_per_interaction)

    if eval_interval and agent_learner.train_step_numpy % eval_interval == 0:
      logging.info('Evaluating.')
      eval_actor.run_and_log()
      for metric in eval_actor.metrics:
        if isinstance(metric, py_metrics.AverageReturnMetric):
          average_returns.append(metric._buffer.mean())  # pylint: disable=protected-access

  # Log last section of evaluation scores for the final metric.
  idx = int(FLAGS.average_last_fraction * len(average_returns))
  avg_return = np.mean(average_returns[-idx:])
  logging.info('Step %d, Average Return: %f', env_step_metric.result(),
               avg_return)

  rb_observer.close()
  reverb_server.stop()


def main(_):
  tf.config.run_functions_eagerly(False)
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.enable_v2_behavior()
  strategy = strategy_utils.get_strategy(FLAGS.tpu, FLAGS.use_gpu)

  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)
  logging.info('Gin bindings: %s', FLAGS.gin_bindings)
  logging.info('# Gin-Config:\n %s', gin.config.operative_config_str())

  train_eval(
      FLAGS.root_dir,
      strategy=strategy,
      reverb_port=FLAGS.reverb_port)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)

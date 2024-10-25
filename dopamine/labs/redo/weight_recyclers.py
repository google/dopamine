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
"""This class implements recycling of dead neurons."""
import functools
import logging
import flax
from flax import linen as nn
import gin
import jax
from jax import random
import jax.numpy as jnp
import optax


def leastk_mask(scores, ones_fraction):
  """Given a tensor of scores creates a binary mask.

  Args:
    scores: top-scores are kept
    ones_fraction: float, of the generated mask.

  Returns:
    array, same shape and type as scores or None.
  """
  if ones_fraction is None or ones_fraction == 0:
    return jnp.zeros_like(scores)
  # This is to ensure indices with smallest values are selected.
  scores = -scores

  n_ones = jnp.round(jnp.size(scores) * ones_fraction)
  k = jnp.maximum(1, n_ones).astype(int)
  flat_scores = jnp.reshape(scores, -1)
  threshold = jax.lax.sort(flat_scores)[-k]

  mask = (flat_scores >= threshold).astype(flat_scores.dtype)
  return mask.reshape(scores.shape)


def reset_momentum(momentum, mask):
  new_momentum = momentum if mask is None else momentum * (1.0 - mask)
  return new_momentum


def weight_reinit_zero(param, mask):
  if mask is None:
    return param
  else:
    new_param = jnp.zeros_like(param)
    param = jnp.where(mask == 1, new_param, param)
    return param


def weight_reinit_random(
    param, mask, key, weight_scaling=False, scale=1.0, weights_type='incoming'
):
  """Randomly reinit recycled weights and may scale its norm.

  If scaling applied, the norm of recycled weights equals
  the average norm of non recycled weights per neuron multiplied by a scalar.

  Args:
    param: current param
    mask: incoming/outgoing mask for recycled weights
    key: random key to generate new random weights
    weight_scaling: if true scale recycled weights with the norm of non recycled
    scale: scale to multiply the new weights norm.
    weights_type: incoming or outgoing weights

  Returns:
  params: new params after weight recycle.
  """
  if mask is None or key is None:
    return param

  new_param = nn.initializers.xavier_uniform()(key, shape=param.shape)

  if weight_scaling:
    axes = list(range(param.ndim))
    if weights_type == 'outgoing':
      del axes[-2]
    else:
      del axes[-1]

    neuron_mask = jnp.mean(mask, axis=axes)

    non_dead_count = neuron_mask.shape[0] - jnp.count_nonzero(neuron_mask)
    norm_per_neuron = _get_norm_per_neuron(param, axes)
    non_recycled_norm = (
        jnp.sum(norm_per_neuron * (1 - neuron_mask)) / non_dead_count
    )
    non_recycled_norm = non_recycled_norm * scale

    normalized_new_param = _weight_normalization_per_neuron_norm(
        new_param, axes
    )
    new_param = normalized_new_param * non_recycled_norm

  param = jnp.where(mask == 1, new_param, param)
  return param


def _weight_normalization_per_neuron_norm(param, axes):
  norm_per_neuron = _get_norm_per_neuron(param, axes)
  norm_per_neuron = jnp.expand_dims(norm_per_neuron, axis=axes)
  normalized_param = param / norm_per_neuron
  return normalized_param


def _get_norm_per_neuron(param, axes):
  return jnp.sqrt(jnp.sum(jnp.power(param, 2), axis=axes))


@gin.configurable
class BaseRecycler:
  """Base class for weight update methods.

  Attributes:
    all_layers_names: list of layer names in a model.
    recycle_type: neuron, layer based.
    dead_neurons_threshold: below this threshold a neuron is considered dead.
    reset_layers: list of layer names to be recycled.
    reset_start_layer_idx: index of the layer from which we start recycling.
    reset_period: int represents the period of weight update.
    reset_start_step: start recycle from start step
    reset_end_step:  end recycle from end step
    logging_period:  the period of statistics logging e.g., dead neurons.
    prev_neuron_score: score at last reset step or log step in case of no reset.
    sub_mean_score: if True the average activation will be subtracted for each
      neuron when we calculate the score.
  """

  def __init__(
      self,
      all_layers_names,
      dead_neurons_threshold=0.0,
      reset_start_layer_idx=0,
      reset_period=200_000,
      reset_start_step=0,
      reset_end_step=100_000_000,
      logging_period=20_000,
      sub_mean_score=False,
  ):
    self.all_layers_names = all_layers_names
    self.dead_neurons_threshold = dead_neurons_threshold
    self.reset_layers = all_layers_names[reset_start_layer_idx:]
    self.reset_period = reset_period
    self.reset_start_step = reset_start_step
    self.reset_end_step = reset_end_step
    self.logging_period = logging_period
    self.prev_neuron_score = None
    self.sub_mean_score = sub_mean_score

  def update_reset_layers(self, reset_start_layer_idx):
    self.reset_layers = self.all_layers_names[reset_start_layer_idx:]

  def is_update_iter(self, step):
    return step > 0 and (step % self.reset_period == 0)

  def update_weights(self, intermediates, params, key, opt_state):
    raise NotImplementedError

  def maybe_update_weights(
      self, update_step, intermediates, params, key, opt_state
  ):
    self._last_update_step = update_step
    if self.is_reset(update_step):
      new_params, new_opt_state = self.update_weights(
          intermediates, params, key, opt_state
      )
    else:
      new_params, new_opt_state = params, opt_state
    return new_params, new_opt_state

  def is_reset(self, update_step):
    del update_step
    return False

  def is_intermediated_required(self, update_step):
    return self.is_logging_step(update_step)

  def is_logging_step(self, step):
    return step % self.logging_period == 0

  def maybe_log_deadneurons(self, update_step, intermediates):
    is_logging = self.is_logging_step(update_step)
    if is_logging:
      return self.log_dead_neurons_count(intermediates)
    else:
      return None

  def intersected_dead_neurons_with_last_reset(
      self, intermediates, update_step
  ):
    if self.is_logging_step(update_step):
      log_dict = self.log_intersected_dead_neurons(intermediates)
      return log_dict
    else:
      return None

  def log_intersected_dead_neurons(self, intermediates):
    """Track intersected dead neurons with last logging/reset step.

    Args:
      intermediates: current intermediates

    Returns:
      log_dict: dict contains the percentage of intersection
    """
    score_tree = jax.tree.map(self.estimate_neuron_score, intermediates)
    neuron_score_dict = flax.traverse_util.flatten_dict(score_tree, sep='/')

    if self.prev_neuron_score is None:
      self.prev_neuron_score = neuron_score_dict
      log_dict = None
    else:
      log_dict = {}
      for prev_k_score, current_k_score in zip(
          self.prev_neuron_score.items(), neuron_score_dict.items()
      ):
        _, prev_score = prev_k_score
        k, score = current_k_score
        prev_score, score = prev_score[0], score[0]
        prev_mask = prev_score <= self.dead_neurons_threshold
        # we count the dead neurons which remains dead in the current step.
        intersected_mask = (prev_mask) & (score <= self.dead_neurons_threshold)
        prev_dead_count = jnp.count_nonzero(prev_mask)
        intersected_count = jnp.count_nonzero(intersected_mask)

        percent = (
            (float(intersected_count) / prev_dead_count)
            if prev_dead_count
            else 0.0
        )
        log_dict[f'dead_intersected_percent/{k[:-9]}'] = float(percent) * 100.0

        # track average score of recycled neurons from last step and
        # the average of non dead in the current step.
        nondead_mask = score > self.dead_neurons_threshold
        log_dict[f'mean_score_recycled/{k[:-9]}'] = float(
            jnp.mean(score[prev_mask])
        )
        log_dict[f'mean_score_nondead/{k[:-9]}'] = float(
            jnp.mean(score[nondead_mask])
        )

      self.prev_neuron_score = neuron_score_dict
    return log_dict

  def log_dead_neurons_count(self, intermediates):
    """log dead neurons in each layer.

    For conv layer we also log dead elements in the spatial dimension.

    Args:
      intermediates: intermidate activation in each layer.

    Returns:
      log_dict_elements_per_neuron
      log_dict_neurons
    """

    def log_dict(score, score_type):
      total_neurons, total_deadneurons = 0.0, 0.0
      score_dict = flax.traverse_util.flatten_dict(score, sep='/')

      log_dict = {}
      for k, m in score_dict.items():
        if 'final_layer' in k:
          continue
        m = m[0]
        layer_size = float(jnp.size(m))
        deadneurons_count = jnp.count_nonzero(m <= self.dead_neurons_threshold)
        total_neurons += layer_size
        total_deadneurons += deadneurons_count
        log_dict[f'dead_{score_type}_percentage/{k[:-9]}'] = (
            float(deadneurons_count) / layer_size
        ) * 100.0
        log_dict[f'dead_{score_type}_count/{k[:-9]}'] = float(deadneurons_count)
      log_dict[f'{score_type}/total'] = total_neurons
      log_dict[f'{score_type}/deadcount'] = float(total_deadneurons)
      log_dict[f'dead_{score_type}_percentage'] = (
          float(total_deadneurons) / total_neurons
      ) * 100.0
      return log_dict

    neuron_score = jax.tree.map(self.estimate_neuron_score, intermediates)
    log_dict_neurons = log_dict(neuron_score, 'feature')

    return log_dict_neurons

  def estimate_neuron_score(self, activation, is_cbp=False):
    """Calculates neuron score based on absolute value of activation.

    The score of feature map is the normalized average score over
    the spatial dimension.

    Args:
      activation: intermediate activation of each layer
      is_cbp: if true, subtracts the mean and skips normalization.

    Returns:
      element_score: score of each element in feature map in the spatial dim.
      neuron_score: score of feature map
    """
    reduce_axes = list(range(activation.ndim - 1))
    if self.sub_mean_score or is_cbp:
      activation = activation - jnp.mean(activation, axis=reduce_axes)

    score = jnp.mean(jnp.abs(activation), axis=reduce_axes)
    if not is_cbp:
      # Normalize so that all scores sum to one.
      score /= jnp.mean(score) + 1e-9

    return score


@gin.configurable
class LayerReset(BaseRecycler):
  """Reset all weights of some layers.

  This class implements the primacy bias paper, in which all weights of the
  last layers of the model are randomly reinitalized.

  Attributes:
  """

  def is_reset(self, update_step):
    within_reset_interval = (
        update_step >= self.reset_start_step
        and update_step < self.reset_end_step
    )
    return self.is_update_iter(update_step) and within_reset_interval

  def update_weights(self, intermediates, params, key, opt_state):
    param_dict = flax.traverse_util.flatten_dict(params, sep='/')
    mask_dict = {
        k: jnp.zeros_like(p) if p.ndim != 1 else None
        for k, p in param_dict.items()
    }
    random_keys_dict = {k: None for k in param_dict}
    # set mask = 1 for the weights that will be recycled
    for param_key, param in param_dict.items():
      param_layer_name = param_key.split('/')[1]
      if param.ndim != 1 and param_layer_name in self.reset_layers:
        mask_dict[param_key] = jnp.ones_like(param)
        key, subkey = random.split(key)
        random_keys_dict[param_key] = subkey
        # reset bias
        bias_key = 'params/' + param_layer_name + '/bias'
        new_bias = jnp.zeros_like(param_dict[bias_key])
        param_dict[bias_key] = new_bias

    params = flax.core.freeze(
        flax.traverse_util.unflatten_dict(param_dict, sep='/')
    )
    masks = flax.core.freeze(
        flax.traverse_util.unflatten_dict(mask_dict, sep='/')
    )
    random_keys = flax.core.freeze(
        flax.traverse_util.unflatten_dict(random_keys_dict, sep='/')
    )
    # reset weights
    weight_random_reset_fn = jax.jit(
        functools.partial(jax.tree.map, weight_reinit_random)
    )
    params = weight_random_reset_fn(params, masks, random_keys)

    # reset mu, nu of adam optimizer for recycled weights.
    reset_momentum_fn = jax.jit(functools.partial(jax.tree.map, reset_momentum))
    new_mu = reset_momentum_fn(opt_state[0][1], masks)
    new_nu = reset_momentum_fn(opt_state[0][2], masks)
    opt_state_list = list(opt_state)
    opt_state_list[0] = optax.ScaleByAdamState(
        opt_state[0].count, mu=new_mu, nu=new_nu
    )
    opt_state = tuple(opt_state_list)
    return params, opt_state


@gin.configurable
class NeuronRecycler(BaseRecycler):
  """Recycle the weights connected to dead neurons.

  In convolutional neural networks, we consider a feature map as neuron.

  Attributes:
    next_layers: dict key a current layer name, value next layer name.
    init_method_outgoing: method to init outgoing weights (random, zero).
    weight_scaling: if true, scale reinit weights.
    incoming_scale: scalar for incoming weights.
    outgoing_scale: scalar for outgoing weights.
  """

  def __init__(
      self,
      all_layers_names,
      init_method_outgoing='zero',
      weight_scaling=False,
      incoming_scale=1.0,
      outgoing_scale=1.0,
      network='nature',
      **kwargs,
  ):
    super(NeuronRecycler, self).__init__(all_layers_names, **kwargs)
    self.init_method_outgoing = init_method_outgoing
    self.weight_scaling = weight_scaling
    self.incoming_scale = incoming_scale
    self.outgoing_scale = outgoing_scale
    # prepare a dict that has pointer to next layer give a layer name
    # this is needed because neuron recycle reinitalizes both sides
    # (incoming and outgoing weights) of a neuron and needs a point to the
    # outgoing weights.
    self.next_layers = {}
    for current_layer, next_layer in zip(
        all_layers_names[:-1], all_layers_names[1:]
    ):
      self.next_layers[current_layer] = next_layer

    # we don't recycle the neurons in the output layer.
    self.reset_layers = self.reset_layers[:-1]

    # if network is resnet, recycle only the incoming/outgoing of the first conv
    # layer in each block and final dense layer
    if network == 'resnet':
      self.reset_layers = []
      for layer in self.all_layers_names:
        if 'Conv_1' in layer or 'Conv_3' in layer or 'Dense' in layer:
          self.reset_layers.append(layer)

  def intersected_dead_neurons_with_last_reset(
      self, intermediates, update_step
  ):
    if self.is_reset(update_step):
      log_dict = self.log_intersected_dead_neurons(intermediates)
      return log_dict
    else:
      return None

  def is_reset(self, update_step):
    within_reset_interval = (
        update_step >= self.reset_start_step
        and update_step < self.reset_end_step
    )
    return self.is_update_iter(update_step) and within_reset_interval

  def is_intermediated_required(self, update_step):
    is_logging = self.is_logging_step(update_step)
    is_update_iter = self.is_update_iter(update_step)
    return is_logging or is_update_iter

  def update_reset_layers(self, reset_start_layer_idx):
    self.reset_layers = self.all_layers_names[reset_start_layer_idx:]
    self.reset_layers = self.reset_layers[:-1]

  def update_weights(self, intermediates, params, key, opt_state):
    new_param, opt_state = self.recycle_dead_neurons(
        intermediates, params, key, opt_state
    )
    return new_param, opt_state

  def recycle_dead_neurons(self, intermedieates, params, key, opt_state):
    """Recycle dead neurons by reinitalizie incoming and outgoing connections.

    Incoming connections are randomly initalized and outgoing connections
    are zero initalized.
    A featuremap is considered dead when its score is below or equal
    dead neuron threshold.
    Args:
      intermedieates: pytree contains the activations over a batch.
      params: current weights of the model.
      key: used to generate random keys.
      opt_state: state of optimizer.

    Returns:
      new model params after recycling dead neurons.
      opt_state: new state for the optimizer

    Raises: raise error if init_method_outgoing is not one of the following
    (random, zero).
    """
    activations_score_dict = flax.traverse_util.flatten_dict(
        intermedieates, sep='/'
    )
    param_dict = flax.traverse_util.flatten_dict(params, sep='/')

    # create incoming and outgoing masks and reset bias of dead neurons.
    (
        incoming_mask_dict,
        outgoing_mask_dict,
        incoming_random_keys_dict,
        outgoing_random_keys_dict,
        param_dict,
    ) = self.create_masks(param_dict, activations_score_dict, key)

    params = flax.core.freeze(
        flax.traverse_util.unflatten_dict(param_dict, sep='/')
    )
    incoming_random_keys = flax.core.freeze(
        flax.traverse_util.unflatten_dict(incoming_random_keys_dict, sep='/')
    )
    if self.init_method_outgoing == 'random':
      outgoing_random_keys = flax.core.freeze(
          flax.traverse_util.unflatten_dict(outgoing_random_keys_dict, sep='/')
      )
    # reset incoming weights
    incoming_mask = flax.core.freeze(
        flax.traverse_util.unflatten_dict(incoming_mask_dict, sep='/')
    )
    reinit_fn = functools.partial(
        weight_reinit_random,
        weight_scaling=self.weight_scaling,
        scale=self.incoming_scale,
        weights_type='incoming',
    )
    weight_random_reset_fn = jax.jit(functools.partial(jax.tree.map, reinit_fn))
    params = weight_random_reset_fn(params, incoming_mask, incoming_random_keys)

    # reset outgoing weights
    outgoing_mask = flax.core.freeze(
        flax.traverse_util.unflatten_dict(outgoing_mask_dict, sep='/')
    )
    if self.init_method_outgoing == 'random':
      reinit_fn = functools.partial(
          weight_reinit_random,
          weight_scaling=self.weight_scaling,
          scale=self.outgoing_scale,
          weights_type='outgoing',
      )
      weight_random_reset_fn = jax.jit(
          functools.partial(jax.tree.map, reinit_fn)
      )
      params = weight_random_reset_fn(
          params, outgoing_mask, outgoing_random_keys
      )
    elif self.init_method_outgoing == 'zero':
      weight_zero_reset_fn = jax.jit(
          functools.partial(jax.tree.map, weight_reinit_zero)
      )
      params = weight_zero_reset_fn(params, outgoing_mask)
    else:
      raise ValueError(f'Invalid init method: {self.init_method_outgoing}')
    # reset mu, nu of adam optimizer for recycled weights.
    reset_momentum_fn = jax.jit(functools.partial(jax.tree.map, reset_momentum))
    new_mu = reset_momentum_fn(opt_state[0][1], incoming_mask)
    new_mu = reset_momentum_fn(new_mu, outgoing_mask)
    new_nu = reset_momentum_fn(opt_state[0][2], incoming_mask)
    new_nu = reset_momentum_fn(new_nu, outgoing_mask)
    opt_state_list = list(opt_state)
    opt_state_list[0] = optax.ScaleByAdamState(
        opt_state[0].count, mu=new_mu, nu=new_nu
    )
    opt_state = tuple(opt_state_list)
    return params, opt_state

  def _score2mask(self, activation, param, next_param, key):
    del key, param, next_param
    score = self.estimate_neuron_score(activation)
    return score <= self.dead_neurons_threshold

  def create_masks(self, param_dict, activations_dict, key):
    """create the masks for recycled weights based on neurons scores.

    Args:
      param_dict: dict of model params.
      activations_dict: dict of the neuron score of each layer.
      key: used seed for random weights.

    Returns:
      incoming_mask_dict
      outgoing_mask_dict
      ingoing_random_keys_dict
      outgoing_random_keys_dict
      param_dict
    """
    incoming_mask_dict = {
        k: jnp.zeros_like(p) if p.ndim != 1 else None
        for k, p in param_dict.items()
    }
    outgoing_mask_dict = {
        k: jnp.zeros_like(p) if p.ndim != 1 else None
        for k, p in param_dict.items()
    }
    ingoing_random_keys_dict = {k: None for k in param_dict}
    outgoing_random_keys_dict = (
        {k: None for k in param_dict}
        if self.init_method_outgoing == 'random'
        else {}
    )

    # prepare mask of incoming and outgoing recycled connections
    for k in self.reset_layers:
      param_key = 'params/' + k + '/kernel'
      param = param_dict[param_key]
      # This won't work for DRQ, since returned keys can be a list.
      # We don't support that at the moment.
      next_key = self.next_layers[k]
      if isinstance(next_key, list):
        next_key = next_key[0]
      next_param = param_dict['params/' + next_key + '/kernel']
      activation = activations_dict[k + '_act/__call__'][0]
      # TODO(evcu) Maybe use per_layer random keys here.
      neuron_mask = self._score2mask(activation, param, next_param, key)

      # the for loop handles the case where a layer has multiple next layers
      # like the case in DrQ where the output layer has multihead.
      next_keys = (
          self.next_layers[k]
          if isinstance(self.next_layers[k], list)
          else [self.next_layers[k]]
      )
      for next_k in next_keys:
        next_param_key = 'params/' + next_k + '/kernel'
        next_param = param_dict[next_param_key]
        incoming_mask, outgoing_mask = self.create_mask_helper(
            neuron_mask, param, next_param
        )
        incoming_mask_dict[param_key] = incoming_mask
        outgoing_mask_dict[next_param_key] = outgoing_mask
        key, subkey = random.split(key)
        ingoing_random_keys_dict[param_key] = subkey
        if self.init_method_outgoing == 'random':
          key, subkey = random.split(key)
          outgoing_random_keys_dict[next_param_key] = subkey

      # reset bias
      bias_key = 'params/' + k + '/bias'
      new_bias = jnp.zeros_like(param_dict[bias_key])
      param_dict[bias_key] = jnp.where(
          neuron_mask, new_bias, param_dict[bias_key]
      )

    return (
        incoming_mask_dict,
        outgoing_mask_dict,
        ingoing_random_keys_dict,
        outgoing_random_keys_dict,
        param_dict,
    )

  def create_mask_helper(self, neuron_mask, current_param, next_param):
    """generate incoming and outgoing weight mask given dead neurons mask.

    Args:
      neuron_mask: mask of size equals the width of a layer.
      current_param: incoming weights of a layer.
      next_param: outgoing weights of a layer.

    Returns:
      incoming_mask
      outgoing_mask
    """

    def mask_creator(expansion_axis, expansion_axes, param, neuron_mask):
      """create a mask of weight matrix given 1D vector of neurons mask.

      Args:
        expansion_axis: List contains 1 axis. The dimension to expand the mask
          for dense layers (weight shape 2D).
        expansion_axes: List conrtains 3 axes. The dimensions to expand the
          score for convolutional layers (weight shape 4D).
        param: weight.
        neuron_mask: 1D mask that represents dead neurons(features).

      Returns:
        mask: mask of weight.
      """
      if param.ndim == 2:
        axes = expansion_axis
        # flatten layer
        # The size of neuron_mask is the same as the width of last conv layer.
        # This conv layer will be flatten and connected to dense layer.
        # we repeat each value of a feature map to cover the spatial dimension.
        if axes[0] == 1 and (param.shape[0] > neuron_mask.shape[0]):
          num_repeatition = int(param.shape[0] / neuron_mask.shape[0])
          neuron_mask = jnp.repeat(neuron_mask, num_repeatition, axis=0)
      elif param.ndim == 4:
        axes = expansion_axes
      mask = jnp.expand_dims(neuron_mask, axis=tuple(axes))
      for i in range(len(axes)):
        mask = jnp.repeat(mask, param.shape[axes[i]], axis=axes[i])
      return mask

    incoming_mask = mask_creator([0], [0, 1, 2], current_param, neuron_mask)
    outgoing_mask = mask_creator([1], [0, 1, 3], next_param, neuron_mask)
    return incoming_mask, outgoing_mask


@gin.configurable
class NeuronRecyclerScheduled(NeuronRecycler):
  """Fixed scheduled version of the NeuronRecycler."""

  def __init__(
      self,
      *args,
      score_type='redo',
      recycle_rate=0.3,
      **kwargs,
  ):
    super(NeuronRecyclerScheduled, self).__init__(*args, **kwargs)
    self.score_type = score_type
    self.recycle_rate = recycle_rate

  def _score2mask(self, activation, param, next_param, key):
    is_cbp = self.score_type == 'cbp'
    score = self.estimate_neuron_score(activation, is_cbp=is_cbp)
    if self.score_type == 'redo':
      pass
    elif self.score_type == 'random':
      new_key = random.fold_in(key, self._last_update_step)
      score = random.permutation(new_key, score, independent=True)
    elif self.score_type == 'redo_inverted':
      score = -score
    # Metric used in Continual Backprop pape.
    elif self.score_type == 'cbp':
      next_axes = list(range(param.ndim))
      del next_axes[-2]
      current_axes = list(range(param.ndim))
      del current_axes[-1]
      if next_param.ndim == 2 and param.ndim == 4:
        new_shape = activation.shape[1:] + (-1,)
        next_param = jnp.reshape(next_param, new_shape)
      score *= jnp.sum(jnp.abs(next_param), axis=next_axes) / jnp.sum(
          jnp.abs(param), axis=current_axes
      )
    multiplier = max(0, self._last_update_step / self.reset_end_step)
    ones_fraction = float(jnp.cos(jnp.pi * 0.5 * multiplier))
    ones_fraction *= self.recycle_rate
    logging.info(
        'score_type: %s, multiplier: %f, ones_fraction=%f',
        self.score_type,
        multiplier,
        ones_fraction,
    )
    return leastk_mask(score, ones_fraction)

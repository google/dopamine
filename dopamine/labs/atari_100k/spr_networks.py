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
"""Networks for SPR in Jax+Dopamine."""

import collections
import functools
import time
from typing import Any, Callable, Optional, Tuple

from dopamine.jax.networks import preprocess_atari_inputs
from flax import linen as nn
import gin
import jax
from jax import lax
from jax import random
from jax.nn import initializers
import jax.numpy as jnp
import numpy as onp

PRNGKey = Any
Array = Any
Shape = Tuple[int]
Dtype = Any


SPROutputType = collections.namedtuple(
    'RL_network', ['q_values', 'logits', 'probabilities', 'latent']
)


def _absolute_dims(rank, dims):
  return tuple([rank + dim if dim < 0 else dim for dim in dims])


# --------------------------- < NoisyNetwork >---------------------------------
# Noisy networks for SPR need to be called multiple times with and without
# noise, so we have a slightly customized implementation where eval_mode
# is an argument to __call__ rather than an attribute of the class.
@gin.configurable
class NoisyNetwork(nn.Module):
  """Noisy Network from Fortunato et al. (2018)."""

  features: int = 512

  @staticmethod
  def sample_noise(key, shape):
    return random.normal(key, shape)

  @staticmethod
  def f(x):
    # See (10) and (11) in Fortunato et al. (2018).
    return jnp.multiply(jnp.sign(x), jnp.power(jnp.abs(x), 0.5))

  @nn.compact
  def __call__(self, x, rng_key, bias=True, kernel_init=None, eval_mode=False):
    """Call the noisy layer.

    Args:
      x: Data point. jnp.float32 tensor, without batch dimension
      rng_key: JAX prng key
      bias: Whether or not to use bias params (static)
      kernel_init: Init function for kernels
      eval_mode: Enable eval mode. Disables noise parameters.

    Returns:
      The transformed output. JNP tensor.
    """

    def mu_init(key, shape):
      # Initialization of mean noise parameters (Section 3.2)
      low = -1 / jnp.power(x.shape[-1], 0.5)
      high = 1 / jnp.power(x.shape[-1], 0.5)
      return random.uniform(key, minval=low, maxval=high, shape=shape)

    def sigma_init(key, shape, dtype=jnp.float32):  # pylint: disable=unused-argument
      # Initialization of sigma noise parameters (Section 3.2)
      return jnp.ones(shape, dtype) * (0.5 / onp.sqrt(x.shape[-1]))

    # Factored gaussian noise in (10) and (11) in Fortunato et al. (2018).
    p = NoisyNetwork.sample_noise(rng_key, [x.shape[-1], 1])
    q = NoisyNetwork.sample_noise(rng_key, [1, self.features])
    f_p = NoisyNetwork.f(p)
    f_q = NoisyNetwork.f(q)
    w_epsilon = f_p * f_q
    b_epsilon = jnp.squeeze(f_q)

    # See (8) and (9) in Fortunato et al. (2018) for output computation.
    w_mu = self.param('kernel', mu_init, (x.shape[-1], self.features))
    w_sigma = self.param('kernell', sigma_init, (x.shape[-1], self.features))
    w_epsilon = jnp.where(
        eval_mode,
        onp.zeros(shape=(x.shape[-1], self.features), dtype=onp.float32),
        w_epsilon,
    )
    w = w_mu + jnp.multiply(w_sigma, w_epsilon)
    ret = jnp.matmul(x, w)

    b_epsilon = jnp.where(
        eval_mode,
        onp.zeros(shape=(self.features,), dtype=onp.float32),
        b_epsilon,
    )
    b_mu = self.param('bias', mu_init, (self.features,))
    b_sigma = self.param('biass', sigma_init, (self.features,))
    b = b_mu + jnp.multiply(b_sigma, b_epsilon)
    return jnp.where(bias, ret + b, ret)


# -------------------------- < RainbowNetwork >---------------------------------


class NoStatsBatchNorm(nn.Module):
  """A version of BatchNorm that does not track running statistics.

  For use in places where this functionality is not available in Jax.
  Attributes:
    axis: the feature or non-batch axis of the input.
    epsilon: a small float added to variance to avoid dividing by zero.
    dtype: the dtype of the computation (default: float32).
    use_bias:  if True, bias (beta) is added.
    use_scale: if True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: initializer for bias, by default, zero.
    scale_init: initializer for scale, by default, one.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See `jax.pmap` for a description of axis names (default: None).
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, `[[0, 1], [2, 3]]` would independently batch-normalize over the
      examples on the first two and last two devices. See `jax.lax.psum` for
      more details.
  """

  use_running_average: Optional[bool] = None
  axis: int = -1
  epsilon: float = 1e-5
  dtype: Dtype = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
  scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
  axis_name: Optional[str] = None
  axis_index_groups: Any = None

  @nn.compact
  def __call__(self, x, use_running_average: Optional[bool] = None):
    """Normalizes the input using batch statistics.

    NOTE:
    During initialization (when parameters are mutable) the running average
    of the batch statistics will not be updated. Therefore, the inputs
    fed during initialization don't need to match that of the actual input
    distribution and the reduction axis (set with `axis_name`) does not have
    to exist.
    Args:
      x: the input to be normalized.
      use_running_average: if true, the statistics stored in batch_stats will be
        used instead of computing the batch statistics on the input.

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    x = jnp.asarray(x, jnp.float32)
    axis = self.axis if isinstance(self.axis, tuple) else (self.axis,)
    axis = _absolute_dims(x.ndim, axis)
    feature_shape = tuple(d if i in axis else 1 for i, d in enumerate(x.shape))
    reduced_feature_shape = tuple(d for i, d in enumerate(x.shape) if i in axis)
    reduction_axis = tuple(i for i in range(x.ndim) if i not in axis)

    # see NOTE above on initialization behavior
    initializing = self.is_mutable_collection('params')

    mean = jnp.mean(x, axis=reduction_axis, keepdims=False)
    mean2 = jnp.mean(lax.square(x), axis=reduction_axis, keepdims=False)
    if self.axis_name is not None and not initializing:
      concatenated_mean = jnp.concatenate([mean, mean2])
      mean, mean2 = jnp.split(
          lax.pmean(
              concatenated_mean,
              axis_name=self.axis_name,
              axis_index_groups=self.axis_index_groups,
          ),
          2,
      )
    var = mean2 - lax.square(mean)

    y = x - mean.reshape(feature_shape)
    mul = lax.rsqrt(var + self.epsilon)
    if self.use_scale:
      scale = self.param(
          'scale', self.scale_init, reduced_feature_shape
      ).reshape(feature_shape)
      mul = mul * scale
    y = y * mul
    if self.use_bias:
      bias = self.param('bias', self.bias_init, reduced_feature_shape).reshape(
          feature_shape
      )
      y = y + bias
    return jnp.asarray(y, self.dtype)


def feature_layer(noisy, features):
  """Network feature layer depending on whether noisy_nets are used on or not."""
  if noisy:
    net = NoisyNetwork(features=features)
  else:
    net = nn.Dense(features, kernel_init=nn.initializers.xavier_uniform())

  def apply(x, key, eval_mode):
    if noisy:
      return net(x, key, True, None, eval_mode)  # pytype: disable=wrong-arg-count
    else:
      return net(x)

  return net, apply


def renormalize(tensor, has_batch=False):
  shape = tensor.shape
  if not has_batch:
    tensor = jnp.expand_dims(tensor, 0)
  tensor = tensor.reshape(tensor.shape[0], -1)
  max_val = jnp.max(tensor, axis=-1, keepdims=True)
  min_val = jnp.min(tensor, axis=-1, keepdims=True)
  return ((tensor - min_val) / (max_val - min_val + 1e-5)).reshape(*shape)


class ConvTMCell(nn.Module):
  """MuZero-style transition model cell, used for SPR.

  Attributes:
    num_actions: how many actions are possible (shape of one-hot vector)
    latent_dim: number of channels in representation.
    renormalize: whether or not to apply renormalization.
  """

  num_actions: int
  latent_dim: int
  renormalize: bool

  def setup(self):
    self.bn = NoStatsBatchNorm(axis=-1, axis_name='batch')

  @nn.compact
  def __call__(self, x, action, eval_mode=False, key=None):
    sizes = [self.latent_dim, self.latent_dim]
    kernel_sizes = [3, 3]
    stride_sizes = [1, 1]

    action_onehot = jax.nn.one_hot(action, self.num_actions)
    action_onehot = jax.lax.broadcast(action_onehot, (x.shape[-3], x.shape[-2]))
    x = jnp.concatenate([x, action_onehot], -1)
    for layer in range(1):
      x = nn.Conv(
          features=sizes[layer],
          kernel_size=(kernel_sizes[layer], kernel_sizes[layer]),
          strides=(stride_sizes[layer], stride_sizes[layer]),
          kernel_init=nn.initializers.xavier_uniform(),
      )(x)
      x = nn.relu(x)
    x = nn.Conv(
        features=sizes[-1],
        kernel_size=(kernel_sizes[-1], kernel_sizes[-1]),
        strides=(stride_sizes[-1], stride_sizes[-1]),
        kernel_init=nn.initializers.xavier_uniform(),
    )(x)
    x = nn.relu(x)

    if self.renormalize:
      x = renormalize(x)

    return x, x


class RainbowCNN(nn.Module):
  """A Jax implementation of the standard 3-layer CNN used in Atari.

  Attributes:
    padding: which padding style to use. Defaults to SAME, which yields larger
      final latents.
  """

  padding: Any = 'SAME'

  stack_sizes: Tuple[int, ...] = (32, 64, 64)

  @nn.compact
  def __call__(self, x):
    # x = x[None, Ellipsis]
    hidden_sizes = self.stack_sizes
    kernel_sizes = [8, 4, 3]
    stride_sizes = [4, 2, 1]
    for layer in range(3):
      x = nn.Conv(
          features=hidden_sizes[layer],
          kernel_size=(kernel_sizes[layer], kernel_sizes[layer]),
          strides=(stride_sizes[layer], stride_sizes[layer]),
          kernel_init=nn.initializers.xavier_uniform(),
          padding=self.padding,
      )(x)
      x = nn.relu(x)  # flatten
    return x


class TransitionModel(nn.Module):
  """A Jax implementation of the SPR transition model, leveraging scan.

  Attributes:
    num_actions: How many possible actions exist.
    latent_dim: Output size.
    renormalize: Whether or not to apply renormalization.
  """

  num_actions: int
  latent_dim: int
  renormalize: bool

  @nn.compact
  def __call__(self, x, action):
    scan = nn.scan(
        ConvTMCell,
        in_axes=0,
        out_axes=0,
        variable_broadcast=['params'],
        split_rngs={'params': False},
    )(
        latent_dim=self.latent_dim,
        num_actions=self.num_actions,
        renormalize=self.renormalize,
    )
    return scan(x, action)


@gin.configurable
class SPRNetwork(nn.Module):
  """Jax Rainbow network for Full Rainbow.

  Attributes:
      num_actions: The number of actions the agent can take at any state.
      num_atoms: The number of buckets of the value function distribution.
      noisy: Whether to use noisy networks.
      dueling: Whether to use dueling network architecture.
      distributional: Whether to use distributional RL.
  """

  num_actions: int
  num_atoms: int
  noisy: bool
  dueling: bool
  distributional: bool
  renormalize: bool = True
  padding: Any = 'SAME'
  inputs_preprocessed: bool = True
  project_relu: bool = False

  def setup(self):
    self.transition_model = TransitionModel(
        num_actions=self.num_actions,
        latent_dim=64,
        renormalize=self.renormalize,
    )
    self.projection, self.apply_projection = feature_layer(self.noisy, 512)
    self.predictor = nn.Dense(512)
    self.encoder = RainbowCNN(stack_sizes=(32, 64, 64))

  def encode(self, x):
    latent = self.encoder(x)
    if self.renormalize:
      latent = renormalize(latent)
    return latent

  def project(self, x, key, eval_mode):
    projected = self.apply_projection(x, key=key, eval_mode=eval_mode)
    if self.project_relu:
      projected = nn.relu(projected)
    return projected

  @functools.partial(jax.vmap, in_axes=(None, 0, None, None))
  def spr_predict(self, x, key, eval_mode):
    projected = self.apply_projection(x, key=key, eval_mode=eval_mode)
    if self.project_relu:
      return nn.relu(self.predictor(nn.relu(projected)))
    else:
      return self.predictor(projected)

  def spr_rollout(self, latent, actions, key):
    _, pred_latents = self.transition_model(latent, actions)
    predictions = self.spr_predict(
        pred_latents.reshape(pred_latents.shape[0], -1), key, True
    )
    return predictions

  @nn.compact
  def __call__(
      self,
      x,
      support,
      actions=None,
      do_rollout=False,
      eval_mode=False,
      key=None,
  ):
    if not self.inputs_preprocessed:
      x = preprocess_atari_inputs(x)

    # Generate a random number generation key if not provided
    if key is None:
      key = random.PRNGKey(int(time.time() * 1e6))

    latent = self.encode(x)
    x = self.apply_projection(
        latent.reshape(-1), key, eval_mode
    )  # Single hidden layer of size 512
    x = nn.relu(x)

    if self.dueling:
      key, rng1, rng2 = random.split(key, 3)
      _, adv_net = feature_layer(self.noisy, self.num_actions * self.num_atoms)
      _, val_net = feature_layer(self.noisy, self.num_atoms)
      adv = adv_net(x, rng1, eval_mode)
      value = val_net(x, rng2, eval_mode)
      adv = adv.reshape((self.num_actions, self.num_atoms))
      value = value.reshape((1, self.num_atoms))
      logits = value + (adv - (jnp.mean(adv, -2, keepdims=True)))
    else:
      key, rng1 = random.split(key, 2)
      _, adv_net = feature_layer(self.noisy, self.num_actions * self.num_atoms)
      x = adv_net(x, rng1, eval_mode)
      logits = x.reshape((self.num_actions, self.num_atoms))

    if do_rollout:
      latent = self.spr_rollout(latent, actions, key)

    if self.distributional:
      probabilities = jnp.squeeze(nn.softmax(logits))
      q_values = jnp.squeeze(jnp.sum(support * probabilities, axis=-1))
      return SPROutputType(q_values, logits, probabilities, latent)

    q_values = jnp.squeeze(logits)
    return SPROutputType(q_values, None, None, latent)

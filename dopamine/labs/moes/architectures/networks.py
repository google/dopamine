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
"""Flax networks that include MoE components."""

import enum
import time
from typing import Any, Tuple
from absl import logging
import chex
from dopamine.jax import networks as base_networks
from dopamine.labs.moes.architectures import moe
from dopamine.labs.moes.architectures import softmoe
from dopamine.labs.moes.architectures import types
import flax.linen as nn
import gin
import jax
from jax import random
import jax.numpy as jnp


@gin.configurable
class ExpertModel(nn.Module):
  """The MLP network of an expert."""

  expert_hidden_size: int
  rng_key: jax.Array
  maintain_token_size: bool = True
  noisy: bool = False
  eval_mode: bool = False
  initializer: Any = nn.initializers.xavier_uniform()

  def setup(self):
    logging.info('\t Creating noisy net: %s', self.noisy)
    if self.noisy:
      self.net = base_networks.NoisyNetwork(
          rng_key=self.rng_key, eval_mode=self.eval_mode
      )
    else:
      self.net = nn.Dense(
          features=self.expert_hidden_size,
          kernel_init=self.initializer,
      )

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    token_size = x.shape[0]

    if self.noisy:
      x = self.net(x, self.expert_hidden_size)
    else:
      x = self.net(x)

    x = nn.relu(x)
    hidden_x = x
    if self.maintain_token_size:
      x = nn.Dense(features=token_size, kernel_init=initializer)(x)
    return x, hidden_x


@gin.configurable
class BigExpertModel(nn.Module):
  """The MLP network of an expert."""

  expert_hidden_size: int
  rng_key: jax.Array
  maintain_token_size: bool = True
  noisy: bool = False
  eval_mode: bool = False
  initializer: Any = nn.initializers.xavier_uniform()
  nn_scale: int = 1
  num_actions: int = 4
  encoder_type: str = 'IMPALA'

  def setup(self):
    logging.info('\t Creating noisy net: %s', self.noisy)
    if self.encoder_type == 'CNN':
      logging.info('\t Creating CNN-encoder ...')
      self.encoder = RainbowCNNet(nn_scale=self.nn_scale)
    elif self.encoder_type == 'IMPALA':
      logging.info('\t Creating IMPALA-encoder ...')
      self.encoder = base_networks.ImpalaEncoder(nn_scale=self.nn_scale)
    if self.noisy:
      self.net = base_networks.NoisyNetwork(
          rng_key=self.rng_key, eval_mode=self.eval_mode
      )
    else:
      self.net = nn.Dense(
          features=self.expert_hidden_size,
          kernel_init=self.initializer,
      )

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    # CNN
    token_size = x.shape[0] * x.shape[1] * x.shape[-1]

    x = self.encoder(x)
    chex.assert_rank(x, 3)

    # Flattened
    x = jnp.reshape(x, -1)

    if self.noisy:
      x = self.net(x, self.expert_hidden_size)
    else:
      x = self.net(x)

    x = nn.relu(x)
    hidden_x = x
    if self.maintain_token_size:
      x = nn.Dense(features=token_size, kernel_init=initializer)(x)

    return x, hidden_x


class RoutingType(enum.Enum):
  # TODO(all) find a better name for per pixel
  # as the token has a size of #channels.
  PER_PIXEL = 'PER_PIXEL'
  PER_FEATUREMAP = 'PER_FEATUREMAP'
  PER_SAMPLE = 'PER_SAMPLE'
  PER_PATCH = 'PER_PATCH'


class MoEType(enum.Enum):
  BASELINE = 'BASELINE'  # Won't use MoE modules.
  MOE = 'MOE'
  SOFTMOE = 'SOFTMOE'
  EXPERTCHOICE = 'EXPERTCHOICE'
  SIMPLICIAL_EMBEDDING_V1 = 'SIMPLICIAL_EMBEDDING_V1'  # Won't use MoE modules.
  SIMPLICIAL_EMBEDDING_V2 = 'SIMPLICIAL_EMBEDDING_V2'  # Won't use MoE modules.


def _maybe_create_moe_module(
    moe_type: str,
    num_features: int,
    num_experts: int,
    num_selected_experts: int,
    routing_type: str,
    noisy: bool,
    rng_key: jax.Array,
    expert_type: str = 'SMALL',
    encoder_type: str = 'IMPALA',
) -> nn.Module | None:
  """Try to create an MoE module, or return None."""
  del routing_type
  moe_type = MoEType[moe_type]
  if moe_type == MoEType.SOFTMOE:
    if expert_type == 'BIG':
      return softmoe.SoftMoE(
          module=BigExpertModel(
              expert_hidden_size=num_features,
              noisy=noisy,
              rng_key=rng_key,
              encoder_type=encoder_type,
          ),
          num_experts=num_experts,
          expert_type=expert_type,
      )
    elif expert_type == 'SMALL':
      return softmoe.SoftMoE(
          module=ExpertModel(
              expert_hidden_size=num_features, noisy=noisy, rng_key=rng_key
          ),
          num_experts=num_experts,
          expert_type=expert_type,
      )
  elif moe_type == MoEType.MOE:
    return moe.MoE(
        module=ExpertModel(
            expert_hidden_size=num_features, noisy=noisy, rng_key=rng_key
        ),
        num_experts=num_experts,
        num_selected_experts=num_selected_experts,
    )
  # Default to baseline/simplicial_embedding.
  return None


@gin.configurable
class ImpalaMoE(nn.Module):
  """The convolutional network used to compute the agent's Q-values."""

  num_actions: int
  inputs_preprocessed: bool = False
  nn_scale: int = 1
  expert_hidden_units: int = 512
  num_experts: int = 8
  num_selected_experts: int = 1
  use_extra_linear_layer: bool = False
  moe_type: str = 'SOFTMOE'
  routing_type: str = 'PER_PIXEL'
  expert_type: str = 'SMALL'
  use_avg_pool_for_tokens: bool = False
  patch_size: Tuple[int, int] = (3, 3)
  embedding_size: int = 32
  scale_hidden_layer: bool = False
  noisy: bool = False
  tau: float = 0.1  # For simplicial embeddings.

  def setup(self):
    assert self.moe_type in MoEType.__members__
    assert self.routing_type in RoutingType.__members__
    self.encoder = base_networks.ImpalaEncoder(nn_scale=self.nn_scale)
    logging.info('\t Creating %s ...', self.__class__.__name__)
    logging.info('\t expert_hidden_units: %s', self.expert_hidden_units)
    logging.info('\t num_experts: %s', self.num_experts)
    logging.info('\t num_selected_experts: %s', self.num_selected_experts)
    logging.info('\t use_extra_linear_layer: %s', self.use_extra_linear_layer)
    logging.info('\t moe_type: %s', self.moe_type)
    logging.info('\t routing_type: %s', self.routing_type)
    logging.info('\t scale_hidden_layer: %s', self.scale_hidden_layer)

  def create_moe(self, rng_key, noisy):
    moe_module = _maybe_create_moe_module(
        moe_type=self.moe_type,
        num_features=self.expert_hidden_units,
        num_experts=self.num_experts,
        num_selected_experts=self.num_selected_experts,
        routing_type=self.routing_type,
        noisy=noisy,
        rng_key=rng_key,
        expert_type=self.expert_type,
    )
    return moe_module

  @nn.compact
  def __call__(self, x: jax.Array, *, key: jax.Array) -> types.NetworkReturn:
    initializer = nn.initializers.xavier_uniform()
    if key is None:
      key = random.PRNGKey(int(time.time() * 1e6))
    if not self.inputs_preprocessed:
      x = base_networks.preprocess_atari_inputs(x)
    if self.expert_type != 'BIG':
      x = self.encoder(x)
    chex.assert_rank(x, 3)

    moe_net = self.create_moe(key, self.noisy)
    if moe_net is None:
      if self.moe_type == 'SIMPLICIAL_EMBEDDING_V1':
        # Simplicial embeddings, from https://arxiv.org/abs/2204.00616.
        x = jax.nn.softmax(x / self.tau, axis=1)
      x = x.reshape((-1))  # flatten
      scale_hidden = (
          self.scale_hidden_layer or self.moe_type == 'SIMPLICIAL_EMBEDDING_V2'
      )
      hidden_features = (
          self.expert_hidden_units * self.num_experts
          if scale_hidden
          else self.expert_hidden_units
      )
      x = nn.Dense(features=hidden_features, kernel_init=initializer)(x)
      if self.moe_type == 'SIMPLICIAL_EMBEDDING_V2':
        # Simplicial embeddings, from https://arxiv.org/abs/2204.00616.
        x = x.reshape((self.num_experts, self.expert_hidden_units))
        x = jax.nn.softmax(x / self.tau, axis=1)
        x = x.reshape((-1))
      x = nn.relu(x)
      x_hidden = x
    else:
      routing_type = RoutingType[self.routing_type]
      if routing_type == RoutingType.PER_FEATUREMAP:
        # For ALE training, the shape of x is currently 121 x 32, which can be
        # interpreted as 121 tokens of dimensionality 32. By setting
        # transpose_tokens to True, we make this 32 tokens of dimensionality
        # 121.
        x = jnp.reshape(x, [x.shape[0] * x.shape[1], x.shape[2]])
        x = jnp.transpose(x)
      elif routing_type == RoutingType.PER_SAMPLE:
        x = x.reshape((1, -1))
      elif routing_type == RoutingType.PER_PATCH:
        if self.use_avg_pool_for_tokens:
          x = nn.avg_pool(
              x,
              window_shape=self.patch_size,
              strides=self.patch_size,
              padding='VALID',
          )
        # Encode patches into tokens of embedding_size
        else:
          x = nn.Conv(
              features=self.embedding_size,
              kernel_size=self.patch_size,
              strides=self.patch_size,
              padding='VALID',
              name='embedding',
          )(x)
        # Reshape images into sequences of tokens.
        x = x.reshape(-1, x.shape[-1])
      elif routing_type == RoutingType.PER_PIXEL:
        x = jnp.reshape(x, [x.shape[0] * x.shape[1], x.shape[2]])
      moe_out = moe_net(x, key=key)
      x = moe_out.values
      x_hidden = moe_out.experts_hidden

    x = x.reshape((-1))  # flatten
    if self.use_extra_linear_layer:
      x = nn.Dense(features=512, kernel_init=initializer)(x)
      x = nn.relu(x)
    q_values = nn.Dense(features=self.num_actions, kernel_init=initializer)(x)
    if moe_net is None:
      return types.BaselineNetworkReturn(q_values=q_values, hidden_act=x_hidden)

    return types.MoENetworkReturn(
        q_values=q_values, moe_out=moe_out, hidden_act=x_hidden
    )


@gin.configurable
class NatureDQNMoE(nn.Module):
  """The Nature CNN network, with an MoE module."""

  num_actions: int
  inputs_preprocessed: bool = False
  nn_scale: int = 1  # Unused in this network.
  expert_hidden_units: int = 512
  num_experts: int = 8
  num_selected_experts: int = 1
  use_extra_linear_layer: bool = False
  moe_type: str = 'SOFTMOE'
  routing_type: str = 'PER_PIXEL'
  expert_type: str = 'SMALL'
  use_avg_pool_for_tokens: bool = False
  patch_size: Tuple[int, int] = (3, 3)
  embedding_size: int = 64
  scale_hidden_layer: bool = False
  noisy: bool = False
  tau: float = 0.1  # For simplicial embeddings.

  def setup(self):
    assert self.moe_type in MoEType.__members__
    assert self.routing_type in RoutingType.__members__
    logging.info('\t Creating %s ...', self.__class__.__name__)
    logging.info('\t expert_hidden_units: %s', self.expert_hidden_units)
    logging.info('\t num_experts: %s', self.num_experts)
    logging.info('\t num_selected_experts: %s', self.num_selected_experts)
    logging.info('\t use_extra_linear_layer: %s', self.use_extra_linear_layer)
    logging.info('\t moe_type: %s', self.moe_type)
    logging.info('\t routing_type: %s', self.routing_type)
    logging.info('\t scale_hidden_layer: %s', self.scale_hidden_layer)

  def create_moe(self, rng_key, noisy):
    moe_module = _maybe_create_moe_module(
        moe_type=self.moe_type,
        num_features=self.expert_hidden_units,
        num_experts=self.num_experts,
        num_selected_experts=self.num_selected_experts,
        routing_type=self.routing_type,
        noisy=noisy,
        rng_key=rng_key,
        expert_type=self.expert_type,
    )
    return moe_module

  @nn.compact
  def __call__(self, x: jax.Array, *, key: jax.Array) -> types.NetworkReturn:
    initializer = nn.initializers.xavier_uniform()
    if key is None:
      key = random.PRNGKey(int(time.time() * 1e6))
    if not self.inputs_preprocessed:
      x = base_networks.preprocess_atari_inputs(x)
    x = nn.Conv(
        features=32, kernel_size=(8, 8), strides=(4, 4), kernel_init=initializer
    )(x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64, kernel_size=(4, 4), strides=(2, 2), kernel_init=initializer
    )(x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64, kernel_size=(3, 3), strides=(1, 1), kernel_init=initializer
    )(x)
    x = nn.relu(x)

    moe_net = self.create_moe(key, self.noisy)
    if moe_net is None:
      if self.moe_type == 'SIMPLICIAL_EMBEDDING_V1':
        # Simplicial embeddings, from https://arxiv.org/abs/2204.00616.
        x = jax.nn.softmax(x / self.tau, axis=1)
      x = x.reshape((-1))  # flatten
      scale_hidden = (
          self.scale_hidden_layer or self.moe_type == 'SIMPLICIAL_EMBEDDING_V2'
      )
      hidden_features = (
          self.expert_hidden_units * self.num_experts
          if scale_hidden
          else self.expert_hidden_units
      )
      x = nn.Dense(features=hidden_features, kernel_init=initializer)(x)
      if self.moe_type == 'SIMPLICIAL_EMBEDDING_V2':
        # Simplicial embeddings, from https://arxiv.org/abs/2204.00616.
        x = x.reshape((self.num_experts, self.expert_hidden_units))
        x = jax.nn.softmax(x / self.tau, axis=1)
        x = x.reshape((-1))
      x = nn.relu(x)
      x_hidden = x
    else:
      routing_type = RoutingType[self.routing_type]
      if routing_type == RoutingType.PER_FEATUREMAP:
        # For ALE training, the shape of x is currently 144 x 64, which can be
        # interpreted as 144 tokens of dimensionality 64. By setting
        # transpose_tokens to True, we make this 64 tokens of dimensionality
        # 144.
        x = jnp.reshape(x, [x.shape[0] * x.shape[1], x.shape[2]])
        x = jnp.transpose(x)
      elif routing_type == RoutingType.PER_SAMPLE:
        x = x.reshape((1, -1))
      elif routing_type == RoutingType.PER_PATCH:
        if self.use_avg_pool_for_tokens:
          x = nn.avg_pool(
              x,
              window_shape=self.patch_size,
              strides=self.patch_size,
              padding='VALID',
          )
        # Encode patches into tokens of embedding_size
        else:
          x = nn.Conv(
              features=self.embedding_size,
              kernel_size=self.patch_size,
              strides=self.patch_size,
              padding='VALID',
              name='embedding',
          )(x)
        # Reshape images into sequences of tokens.
        x = x.reshape(-1, x.shape[-1])
      elif routing_type == RoutingType.PER_PIXEL:
        x = jnp.reshape(x, [x.shape[0] * x.shape[1], x.shape[2]])
      moe_out = moe_net(x, key=key)
      x = moe_out.values
      x_hidden = moe_out.experts_hidden

    x = x.reshape((-1))  # flatten
    if self.use_extra_linear_layer:
      x = nn.Dense(features=512, kernel_init=initializer)(x)
      x = nn.relu(x)
    q_values = nn.Dense(features=self.num_actions, kernel_init=initializer)(x)
    if moe_net is None:
      return types.BaselineNetworkReturn(q_values=q_values, hidden_act=x_hidden)

    return types.MoENetworkReturn(
        q_values=q_values, moe_out=moe_out, hidden_act=x_hidden
    )


@gin.configurable
class RainbowCNNet(nn.Module):
  """Rainbow-CNN encoder."""

  hidden_sizes = (32, 64, 64)
  nn_scale: float = 1

  @nn.compact
  def __call__(self, x):
    kernel_sizes = [8, 4, 3]
    stride_sizes = [4, 2, 1]
    logging.info(
        (
            'Creating CNN network with %s kernel_sizes, %s hidden_sizes, %s'
            ' stride_sizes, nn_scale %s'
        ),
        str(kernel_sizes),
        str(self.hidden_sizes),
        str(stride_sizes),
        str(self.nn_scale),
    )
    for layer in range(3):
      x = nn.Conv(
          features=int(self.hidden_sizes[layer] * self.nn_scale),
          kernel_size=(kernel_sizes[layer], kernel_sizes[layer]),
          strides=(stride_sizes[layer], stride_sizes[layer]),
          kernel_init=nn.initializers.xavier_uniform(),
      )(x)
      x = nn.relu(x)
    return x


@gin.configurable
class FullRainbowMoENetwork(nn.Module):
  """Jax Rainbow network for Full Rainbow with MoEs."""

  num_actions: int
  num_atoms: int
  noisy: bool = True
  dueling: bool = True
  distributional: bool = True
  inputs_preprocessed: bool = False
  expert_hidden_units: int = 512
  num_experts: int = 8
  num_selected_experts: int = 1
  use_extra_linear_layer: bool = False
  moe_type: str = 'SOFTMOE'
  routing_type: str = 'PER_PIXEL'
  expert_type: str = 'SMALL'
  use_avg_pool_for_tokens: bool = False
  patch_size: Tuple[int, int] = (3, 3)
  embedding_size: int = 64
  scale_hidden_layer: bool = False
  encoder_type: str = 'IMPALA'
  nn_scale: float = 1.0
  tau: float = 0.1  # For simplicial embeddings.

  def setup(self):
    logging.info('\t Creating %s ...', self.__class__.__name__)
    logging.info('\t use_extra_linear_layer: %s', self.use_extra_linear_layer)
    logging.info('\t noisy: %s', self.noisy)
    logging.info('\t moe_type: %s', self.moe_type)
    logging.info('\t routing_type: %s', self.routing_type)
    logging.info('\t scale_hidden_layer: %s', self.scale_hidden_layer)
    logging.info('\t encoder_type: %s', self.encoder_type)
    logging.info('\t nn_scale: %s', self.nn_scale)
    logging.info('\t num_actions: %s', self.num_actions)
    logging.info('\t num_atoms: %s', self.num_atoms)
    logging.info('\t num_experts: %s', self.num_experts)
    logging.info('\t expert_hidden_units: %s', self.expert_hidden_units)
    logging.info('\t num_selected_experts: %s', self.num_selected_experts)

    if self.encoder_type == 'CNN':
      logging.info('\t Creating CNN-encoder ...')
      self.encoder = RainbowCNNet(nn_scale=self.nn_scale)
    elif self.encoder_type == 'IMPALA':
      logging.info('\t Creating IMPALA-encoder ...')
      self.encoder = base_networks.ImpalaEncoder(nn_scale=self.nn_scale)

  def create_moe(self, rng_key, noisy):
    moe_module = _maybe_create_moe_module(
        moe_type=self.moe_type,
        num_features=self.expert_hidden_units,
        num_experts=self.num_experts,
        num_selected_experts=self.num_selected_experts,
        routing_type=self.routing_type,
        noisy=noisy,
        rng_key=rng_key,
        expert_type=self.expert_type,
        encoder_type=self.encoder_type,
    )
    return moe_module

  @nn.compact
  def __call__(self, x, support, eval_mode=False, key=None):
    initializer = nn.initializers.xavier_uniform()
    # Generate a random number generation key if not provided
    if key is None:
      key = random.PRNGKey(int(time.time() * 1e6))
    key_1, key_2, key_3 = random.split(key, 3)

    if not self.inputs_preprocessed:
      x = base_networks.preprocess_atari_inputs(x)
    if self.expert_type == 'SMALL':
      x = self.encoder(x)

    net = base_networks.feature_layer(key_1, self.noisy, eval_mode=eval_mode)
    moe_net = self.create_moe(key_2, self.noisy)

    if moe_net is None:
      if self.moe_type == 'SIMPLICIAL_EMBEDDING_V1':
        # Simplicial embeddings, from https://arxiv.org/abs/2204.00616.
        x = jax.nn.softmax(x / self.tau, axis=1)
      x = x.reshape((-1))  # flatten
      scale_hidden = (
          self.scale_hidden_layer or self.moe_type == 'SIMPLICIAL_EMBEDDING_V2'
      )
      hidden_features = (
          self.expert_hidden_units * self.num_experts
          if scale_hidden
          else self.expert_hidden_units
      )
      x = net(x, hidden_features)
      if self.moe_type == 'SIMPLICIAL_EMBEDDING_V2':
        # Simplicial embeddings, from https://arxiv.org/abs/2204.00616.
        x = x.reshape((self.num_experts, self.expert_hidden_units))
        x = jax.nn.softmax(x / self.tau, axis=1)
        x = x.reshape((-1))
      x = nn.relu(x)
    else:
      if x.ndim == 4 and x.shape[0] == 1:
        x = x.squeeze(axis=0)
      chex.assert_rank(x, 3)
      routing_type = RoutingType[self.routing_type]
      if routing_type == RoutingType.PER_FEATUREMAP:
        # For ALE training, the shape of x is currently 144 x 64, which can be
        # interpreted as 144 tokens of dimensionality 64. By setting
        # transpose_tokens to True, we make this 64 tokens of dimensionality
        # 144.
        x = jnp.reshape(x, [x.shape[0] * x.shape[1], x.shape[2]])
        x = jnp.transpose(x)
      elif routing_type == RoutingType.PER_SAMPLE:
        x = x.reshape((1, -1))
      elif routing_type == RoutingType.PER_PATCH:
        if self.use_avg_pool_for_tokens:
          x = nn.avg_pool(
              x,
              window_shape=self.patch_size,
              strides=self.patch_size,
              padding='VALID',
          )
        # Encode patches into tokens of embedding_size
        else:
          x = nn.Conv(
              features=self.embedding_size,
              kernel_size=self.patch_size,
              strides=self.patch_size,
              padding='VALID',
              name='embedding',
          )(x)
        # Reshape images into sequences of tokens.
        x = x.reshape(-1, x.shape[-1])
      elif routing_type == RoutingType.PER_PIXEL:
        x = jnp.reshape(x, [x.shape[0] * x.shape[1], x.shape[2]])
      else:
        raise ValueError(f'Invalid routing type: {self.routing_type}')
      moe_out = moe_net(x, key=key_3)
      x = moe_out.values
      x_hidden = moe_out.experts_hidden

      x = x.reshape((-1))  # flatten

    if self.use_extra_linear_layer:
      x = nn.Dense(features=512, kernel_init=initializer)(x)
      x = nn.relu(x)

    if self.dueling:
      adv = net(x, self.num_actions * self.num_atoms)
      value = net(x, self.num_atoms)
      adv = adv.reshape((self.num_actions, self.num_atoms))
      value = value.reshape((1, self.num_atoms))
      logits = value + (adv - (jnp.mean(adv, axis=0, keepdims=True)))
    else:
      x = net(x, self.num_actions * self.num_atoms)
      logits = x.reshape((self.num_actions, self.num_atoms))

    if self.distributional:
      probabilities = nn.softmax(logits)
      q_values = jnp.sum(support * probabilities, axis=1)
    else:
      probabilities = None
      q_values = jnp.sum(logits, axis=1)  # Sum over all the num_atoms

    if moe_net is None:
      return types.BaselineNetworkReturn(
          q_values=q_values,
          hidden_act=x_hidden,
          logits=logits,
          probabilities=probabilities,
      )

    return types.MoENetworkReturn(
        q_values=q_values,
        logits=logits,
        probabilities=probabilities,
        moe_out=moe_out,
        hidden_act=x_hidden,
    )

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
"""A collection of routers for MoE training."""

from absl import logging
import chex
from dopamine.labs.moes.architectures import types
import flax.linen as nn
import gin
import jax


@gin.configurable
class RandomRouter(nn.Module):
  """Route tokens randomly."""

  num_experts: int | None = None
  k: int = 1

  def setup(self):
    logging.info("Creating a %s", self.__class__.__name__)

  @nn.compact
  def __call__(
      self,
      x: jax.Array,
      *,
      num_experts: int | None = None,
      k: int | None = None,
      route_tokens: int | None = None,
      key: jax.Array | None = None
  ) -> types.RouterReturn:
    chex.assert_rank(x, 2)

    num_experts = nn.merge_param("num_experts", num_experts, self.num_experts)
    k = nn.merge_param("k", k, self.k)
    sequence_length = x.shape[0]

    # probs are set randomly.
    probs = jax.random.normal(key, (sequence_length, num_experts))
    top_expert_weights, top_experts = jax.lax.top_k(probs, k=k)

    return types.RouterReturn(
        output=x,
        probabilities=probs,
        top_expert_weights=top_expert_weights,
        top_experts=top_experts,
    )


@gin.configurable
class TopKRouter(nn.Module):
  """A simple router that linearly projects assignments."""

  k: int
  num_experts: int | None = None
  noisy_routing: bool = False
  noise_std: float = 1.0

  def setup(self):
    logging.info("Creating a %s", self.__class__.__name__)

  @nn.compact
  def __call__(
      self,
      x: jax.Array,
      *,
      num_experts: int | None = None,
      k: int | None = None,
      key: jax.Array | None = None,
      **kwargs
  ) -> types.RouterReturn:
    chex.assert_rank(x, 2)

    num_experts = nn.merge_param("num_experts", num_experts, self.num_experts)
    k = nn.merge_param("k", k, self.k)
    sequence_length = x.shape[0]

    x = nn.Dense(num_experts, use_bias=False)(x)
    chex.assert_shape(x, (sequence_length, num_experts))

    if not self.noisy_routing:
      probs = jax.nn.softmax(x, axis=-1)
    else:
      # A trick proposed by
      # "Scaling Vision with Sparse Mixture of Experts", Riquelme et al., 2021.
      # (https://arxiv.org/abs/2106.05974)
      noise_std = (1.0 / num_experts) * self.noise_std
      noise = noise_std * jax.random.normal(key=key, shape=x.shape)
      x_noisy = x + noise
      probs = jax.nn.softmax(x_noisy, axis=-1)

    top_expert_weights, top_experts = jax.lax.top_k(probs, k=k)

    return types.RouterReturn(
        output=x,
        probabilities=probs,
        top_expert_weights=top_expert_weights,
        top_experts=top_experts,
    )

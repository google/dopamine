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
"""Implementation of SoftMoE for Dopamine networks."""

import math
import chex
from dopamine.labs.moes.architectures import types
import flax.linen as nn
import gin
import jax
import jax.numpy as jnp


def l2_normalize(x, axis, eps=1e-6):
  norm = jnp.sqrt(jnp.square(x).sum(axis=axis, keepdims=True))
  return x * jnp.reciprocal(norm + eps)


@gin.configurable
class SoftMoE(nn.Module):
  """Soft Mixture of Experts (https://arxiv.org/abs/2308.00951)."""

  module: nn.Module
  num_experts: int
  capacity_factor: float = 1.0
  kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
  expert_type: str = "SMALL"
  normalization: bool = False
  use_random_phi: bool = False

  @nn.compact
  def __call__(self, x: jax.Array, *, key: jax.Array) -> types.MoEModuleReturn:
    chex.assert_rank(x, 2)

    # Create phi weight matrix of size (d x n.p), where d is the token dim,
    # n is the number of experts and p is the capacity of each expert (#slots).
    # TODO(gsokar) implementation detail missing. Normalize input and weight
    # The paper states that it make a difference for large tokens
    num_tokens = x.shape[0]
    token_length = x.shape[-1]
    # capacity of each expert
    # we use ceil to allow for per sample token, where the capacity will be 1.
    if self.expert_type == "BIG":
      num_slots = int(
          math.ceil(num_tokens * self.capacity_factor / self.num_experts)
      )
      num_slots_sqrt = math.floor(math.sqrt(num_slots))
      num_slots = int(num_slots_sqrt**2)
    else:
      num_slots = int(
          math.ceil(num_tokens * self.capacity_factor / self.num_experts)
      )
    if self.use_random_phi:
      key, _ = jax.random.split(key)
      phi_weights = jax.random.normal(
          key, (token_length, self.num_experts, num_slots)
      )
    else:
      phi_weights = self.param(
          "phi_weights",
          self.kernel_init,
          (token_length, self.num_experts, num_slots),
      )
    scale_value = self.param("scalar", nn.initializers.ones, (1,))
    # Calculate the weight of each token per slot.
    if self.normalization:
      x_normalized = l2_normalize(x, axis=1)
      phi_weights = scale_value[jnp.newaxis, jnp.newaxis, :].repeat(
          phi_weights.shape[0], axis=0
      ).repeat(phi_weights.shape[1], axis=1).repeat(
          phi_weights.shape[2], axis=2
      ) * l2_normalize(
          phi_weights, axis=0
      )
    else:
      x_normalized = x

    logits = jnp.einsum(
        "md,dnp->mnp",
        x_normalized,
        phi_weights,
    )

    dispatch_weights = jax.nn.softmax(logits, axis=0)
    combine_weights = jax.nn.softmax(logits, axis=(1, 2))

    # Calculate the input tokens to the experts.
    mixture_inputs = jnp.einsum("md,mnp->npd", x, dispatch_weights)
    # Make sure to convert out-of-bounds nans to zeros
    mixture_inputs = jnp.nan_to_num(mixture_inputs)

    if self.expert_type == "BIG":
      dim = int(math.sqrt(num_slots))
      mixture_inputs = mixture_inputs.reshape(
          self.num_experts, 1, dim, dim, token_length
      )

    # Forward pass the MoE
    # This part is taken from MOE class.
    # The input shape should be (num_experts, max_capacity, -1)
    # nn.vmap will map over num_experts without parameter sharing,
    # i.e., it'll use a different initialization for each expert.
    # From there it'll vmap the model over the `max_capacity` dimension.
    experts, experts_hidden = nn.vmap(
        lambda module, x: jax.vmap(module)(x),
        in_axes=0,
        out_axes=0,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        axis_size=self.num_experts,
        # TODO(jfarebro): Supply logical sharding axes
    )(self.module, mixture_inputs)

    if self.expert_type == "BIG":
      experts = experts.reshape(self.num_experts, num_slots, token_length)

    # The output tokens are weighted average of all slots.
    outputs = jnp.einsum("npd,mnp->md", experts, combine_weights)

    probabilities = combine_weights.mean(axis=-1)
    router_out = types.RouterReturn(
        output=jnp.empty_like(probabilities),
        probabilities=probabilities,
        top_expert_weights=jnp.empty([1]),
        top_experts=jnp.argmax(
            combine_weights.mean(axis=-1), axis=1, keepdims=True
        ),
    )
    return types.MoEModuleReturn(
        values=outputs,
        router_out=router_out,
        experts_hidden=experts_hidden,
    )

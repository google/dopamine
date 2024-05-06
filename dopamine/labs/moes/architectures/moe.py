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
"""Basic MoEs for Dopamine networks(https://arxiv.org/abs/2211.15841)."""

import functools
import math

import chex
from dopamine.labs.moes.architectures import routers
from dopamine.labs.moes.architectures import types
import flax.linen as nn
import gin
import jax
import jax.numpy as jnp


@gin.configurable
class MoE(nn.Module):
  """A basic Mixture-of-Experts module (https://arxiv.org/abs/2211.15841)."""

  # TODO(jfarebro): Properly handle masking, what should be done here...?
  # Zeroing out might be fine

  module: nn.Module
  num_experts: int
  num_selected_experts: int
  capacity_factor: float = 1.0
  router_cls: nn.Module = routers.TopKRouter

  def setup(self):
    self.router = self.router_cls(k=self.num_selected_experts)

  @nn.compact
  def __call__(self, x: jax.Array, *, key: jax.Array) -> types.MoEModuleReturn:
    chex.assert_rank(x, 2)
    num_tokens = x.shape[0]
    # we use ceil to allow for per sample token, where the capacity will be 1.
    max_capacity = int(
        math.ceil(num_tokens * self.capacity_factor / self.num_experts)
    )
    # Step 1, Router
    router_out = self.router(x, num_experts=self.num_experts, key=key)

    # Step 2, permutation step.
    #
    # First we'll take advantage of broadcasting to create a matrix
    # where each row is an expert and each column is whether that
    # token is assigned to this expert.
    # TODO(jfarebro): There's probably a better way to do this...
    # It seems needlessly complex and consumes too much memory.
    # At the very least we should be more explicit about the broadcasting here.
    expert_indices = jnp.arange(self.num_experts) == router_out.top_experts
    # We can now use `jnp.nonzero` with the size parameter
    # to get the first `k` assignments and default to some fill value
    # if we can't make up `k` assignments.
    (clipped_expert_indices,) = jax.vmap(
        functools.partial(
            jnp.nonzero,
            size=max_capacity,
            fill_value=num_tokens,  # Fill with an out-of-bound index
        ),
        in_axes=1,
        out_axes=0,
    )(expert_indices)

    # Now we have an array that's (num_experts, max_capacity).
    # We can simply construct a per-expert batch
    # by taking the examples corresponding to the expert indices.
    mixture_inputs = jnp.take_along_axis(
        x[None, ...],
        clipped_expert_indices[..., None],
        axis=1,
        mode=jax.lax.GatherScatterMode.FILL_OR_DROP,
    )
    # Make sure to convert out-of-bounds nans to zeros
    mixture_inputs = jnp.nan_to_num(mixture_inputs)

    # Step 3, forward pass the MoE
    #
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

    expert_output_dims = experts.shape[-1]

    # Step 4: Reverse permutation
    #
    # Now we must reverse the permutation step.
    # To do this we'll use the .at helper around lax.scatter.
    # We can create our output array that's (num_tokens, -1)
    # and scatter the result from the MoE computation.
    # TODO(jfarebro): It'd be really nice to have an implementation of
    # `jnp.put_along_axis`, it would make the code much more readable
    # as `jnp.put_along_axis` should be the inverse of `jnp.take_along_axis`.
    outputs = jnp.zeros((num_tokens, expert_output_dims))
    outputs = outputs.at[clipped_expert_indices.reshape(-1)].set(
        experts.reshape(-1, expert_output_dims),
        mode="drop",  # Drop all out-of-bounds assignments
    )

    return types.MoEModuleReturn(
        values=outputs * router_out.top_expert_weights,
        router_out=router_out,
        experts_hidden=experts_hidden,
    )

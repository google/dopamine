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
"""Return types used by MoE architectures."""

import dataclasses
import jax


@dataclasses.dataclass
class RouterReturn:
  output: jax.Array
  probabilities: jax.Array
  top_expert_weights: jax.Array
  top_experts: jax.Array


def router_flatten(v):
  """Flattening recipe for RouterReturn."""
  children = (v.output, v.probabilities, v.top_expert_weights, v.top_experts)
  aux_data = None
  return (children, aux_data)


def router_unflatten(aux_data, children):
  """Unflattening recipe for RouterReturn."""
  del aux_data
  return RouterReturn(*children)


jax.tree_util.register_pytree_node(
    RouterReturn, router_flatten, router_unflatten
)


@dataclasses.dataclass
class MoEModuleReturn:
  values: jax.Array
  router_out: RouterReturn
  experts_hidden: jax.Array | None = None


def module_flatten(v):
  """Flattening recipe for MoEModuleReturn."""
  children = (v.values, v.router_out, v.experts_hidden)
  aux_data = None
  return (children, aux_data)


def module_unflatten(aux_data, children):
  """Unflattening recipe for MoEModuleReturn."""
  del aux_data
  return MoEModuleReturn(*children)


jax.tree_util.register_pytree_node(
    MoEModuleReturn, module_flatten, module_unflatten
)


@dataclasses.dataclass
class MoENetworkReturn:
  q_values: jax.Array
  moe_out: MoEModuleReturn
  logits: jax.Array | None = None
  probabilities: jax.Array | None = None
  hidden_act: jax.Array | None = None


def network_flatten(v):
  """Flattening recipe for MoENetworkReturn."""
  children = (v.q_values, v.moe_out, v.logits, v.probabilities, v.hidden_act)
  aux_data = None
  return (children, aux_data)


def network_unflatten(aux_data, children):
  """Unflattening recipe for MoENetworkReturn."""
  del aux_data
  return MoENetworkReturn(*children)


jax.tree_util.register_pytree_node(
    MoENetworkReturn, network_flatten, network_unflatten
)


@dataclasses.dataclass
class BaselineNetworkReturn:
  q_values: jax.Array
  hidden_act: jax.Array
  logits: jax.Array | None = None
  probabilities: jax.Array | None = None


def baseline_network_flatten(v):
  """Flattening recipe for BaselineNetworkReturn."""
  children = (v.q_values, v.hidden_act, v.logits, v.probabilities)
  aux_data = None
  return (children, aux_data)


def baseline_network_unflatten(aux_data, children):
  """Unflattening recipe for BaselineNetworkReturn."""
  del aux_data
  return BaselineNetworkReturn(*children)


jax.tree_util.register_pytree_node(
    BaselineNetworkReturn, baseline_network_flatten, baseline_network_unflatten
)


NetworkReturn = MoENetworkReturn | BaselineNetworkReturn

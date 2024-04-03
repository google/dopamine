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
"""Data types used for MoE losses."""

import dataclasses
from typing import Iterable
from dopamine.labs.moes.architectures import types as arch_types
import jax


# We map strings to integers, as str types do not play well with jitted
# functions.
NAME_TO_ID = {
    'EntropyTerm': 0,
    'EntropyLoss': 1,
    'EntropyWeight': 2,
    'ExpertBins': 3,
    'ImportanceTerm': 4,
    'ImportanceLoss': 5,
    'ImportanceWeight': 6,
    'LoadTerm': 7,
    'LoadLoss': 8,
    'LoadWeight': 9,
}
ID_TO_NAME = {v: k for k, v in NAME_TO_ID.items()}
ID_TO_TYPE = {
    1: 'str',
    2: 'histogram',
}


@dataclasses.dataclass
class MoELossParameters:
  """This contains all possible parameters for MoE losses."""

  moe_out: arch_types.MoEModuleReturn
  num_experts: int
  num_selected_experts: int
  key: jax.Array
  entropy_weight: float = 0.5
  importance_weight: float = 1.0
  std_scale: float = 1.0
  load_weight: float = 1.0


def loss_params_flatten(v):
  """Flattening recipe for MoELossParameters."""
  children = (
      v.net_outputs,
      v.num_experts,
      v.num_selected_experts,
      v.key,
      v.std_scale,
      v.load_weight,
  )
  aux_data = None
  return (children, aux_data)


def loss_params_unflatten(aux_data, children):
  """Unflattening recipe for MoELossParameters."""
  del aux_data
  return MoELossParameters(*children)


jax.tree_util.register_pytree_node(
    MoELossParameters, loss_params_flatten, loss_params_unflatten
)


@dataclasses.dataclass
class MoELossStatistic:
  name_id: int
  value: float
  type_id: int = 1


def loss_stat_flatten(v):
  """Flattening recipe for MoELossStatistic."""
  children = (v.name_id, v.value, v.type_id)
  aux_data = None
  return (children, aux_data)


def loss_stat_unflatten(aux_data, children):
  """Unflattening recipe for MoELossStatistic."""
  del aux_data
  return MoELossStatistic(*children)


jax.tree_util.register_pytree_node(
    MoELossStatistic, loss_stat_flatten, loss_stat_unflatten
)


@dataclasses.dataclass
class MoELossReturn:
  value: jax.Array
  statistics: Iterable[MoELossStatistic]


def loss_return_flatten(v):
  """Flattening recipe for MoELossReturn."""
  children = (v.value, v.statistics)
  aux_data = None
  return (children, aux_data)


def loss_return_unflatten(aux_data, children):
  """Unflattening recipe for MoELossReturn."""
  del aux_data
  return MoELossReturn(*children)


jax.tree_util.register_pytree_node(
    MoELossReturn, loss_return_flatten, loss_return_unflatten
)

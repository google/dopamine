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
"""Various losses for training MoEs."""

import functools
from typing import Any, Iterable

from dopamine.labs.moes.agents import types
import gin
import jax
import jax.numpy as jnp


EPS = 1e-6  # To avoid numerical issues with log(0).


def entropy(x):
  x += EPS
  x /= jnp.sum(x)
  # We normalize by log(num_experts) to have values bounded by 1.0.
  return -jnp.sum(x * jnp.log(x)) / jnp.log(x.shape[0])


@gin.configurable
def naive_entropy(
    loss_parameters: types.MoELossParameters,
) -> types.MoELossReturn:
  """A naive entropy loss for load-balancing of experts.

  Simple entropy loss that uses expert bin counts for computation.

  Args:
    loss_parameters: Parameters used for computing the loss.

  Returns:
    Loss value and statistics.
  """
  top_experts = jnp.squeeze(loss_parameters.moe_out.router_out.top_experts)

  def static_bincount(x):
    return jnp.bincount(x, length=loss_parameters.num_experts)

  expert_bins = jax.vmap(static_bincount)(top_experts)
  entropy_term = jnp.mean(jax.vmap(entropy)(expert_bins))
  flattened_bin_counts = jnp.bincount(
      top_experts.flatten(), length=loss_parameters.num_experts
  )
  # We know entropy_term is bounded by 1.0.
  entropy_loss = loss_parameters.entropy_weight * (1.0 - entropy_term)
  statistics = [
      types.MoELossStatistic(
          name_id=types.NAME_TO_ID['EntropyTerm'],  # pytype: disable=wrong-arg-types  # jnp-type
          value=entropy_term,
      ),
      types.MoELossStatistic(
          name_id=types.NAME_TO_ID['EntropyLoss'],  # pytype: disable=wrong-arg-types  # jnp-type
          value=entropy_loss,
      ),
      types.MoELossStatistic(
          name_id=types.NAME_TO_ID['EntropyWeight'],
          value=loss_parameters.entropy_weight,
      ),
      types.MoELossStatistic(
          name_id=types.NAME_TO_ID['ExpertBins'],  # pytype: disable=wrong-arg-types  # jnp-type
          value=flattened_bin_counts,
          type_id=2,
      ),
  ]
  return types.MoELossReturn(value=entropy_loss, statistics=statistics)


@gin.configurable
def importance_loss(
    loss_parameters: types.MoELossParameters,
) -> types.MoELossReturn:
  """Importance loss for load balancing of experts.

  The auxiliary loss defined in Appendix A.2, from
  "Scaling Vision with Sparse Mixture of Experts", Riquelme et al., 2021.
  (https://arxiv.org/abs/2106.05974)
  the code is adapted from
  https://github.com/google-research/vmoe/blob/main/vmoe/nn/routing.py

  Args:
    loss_parameters: Parameters used for computing the loss.

  Returns:
    Loss value and statistics.
  """

  def calculate_importance(gates_softmax):
    axis = tuple(range(gates_softmax.ndim - 1))  # All except last.
    importance_per_expert = jnp.sum(gates_softmax, axis=axis)
    std_importance_per_expert = jnp.std(importance_per_expert)
    mean_importance_per_expert = jnp.mean(importance_per_expert)
    # Compute coefficient of variation (i.e. std/mean) squared.
    return (std_importance_per_expert / mean_importance_per_expert) ** 2

  gates_softmax = loss_parameters.moe_out.router_out.probabilities
  importance_term = jnp.mean(jax.vmap(calculate_importance)(gates_softmax))
  weighted_importance_loss = loss_parameters.importance_weight * importance_term

  statistics = [
      types.MoELossStatistic(
          name_id=types.NAME_TO_ID['ImportanceTerm'],  # pytype: disable=wrong-arg-types  # jnp-type
          value=importance_term,
      ),
      types.MoELossStatistic(
          name_id=types.NAME_TO_ID['ImportanceLoss'],  # pytype: disable=wrong-arg-types  # jnp-type
          value=weighted_importance_loss,
      ),
      types.MoELossStatistic(
          name_id=types.NAME_TO_ID['ImportanceWeight'],
          value=loss_parameters.importance_weight,
      ),
  ]
  return types.MoELossReturn(
      value=weighted_importance_loss, statistics=statistics
  )


@gin.configurable
def load_loss(loss_parameters: types.MoELossParameters) -> types.MoELossReturn:
  """Auxiliary loss for load balancing.

  the auxiliary loss defined in Appendix A.2, from
  "Scaling Vision with Sparse Mixture of Experts", Riquelme et al., 2021.
  (https://arxiv.org/abs/2106.05974)
  the code is adapted from
  https://github.com/google-research/vmoe/blob/main/vmoe/nn/routing.py

  Args:
    loss_parameters: Parameters used for computing the loss.

  Returns:
    Loss value and statistics.
  """

  def calculate_load_loss(
      logits, logits_noisy, noise_std, num_selected_experts
  ):
    del num_selected_experts
    # For each example, compute the weight required for an expert to be selected
    # among the top-k.
    # NOTE: DO NOT TRY TO SIMPLIFY THIS. This convoluted way of obtaining the
    # threshold_per_item avoids adding all-gather ops during backpropagation.
    threshold_per_item_index = jax.lax.top_k(
        logits_noisy, loss_parameters.num_selected_experts
    )[-1][..., -1]
    threshold_per_item = jnp.sum(
        jax.nn.one_hot(threshold_per_item_index, loss_parameters.num_experts)
        * logits_noisy,
        axis=-1,
    )
    # For each example and expert, find how far they were from the threshold and
    # normalize this value by the noise_std to use the standard Gaussian CDF.
    noise_required_to_win = threshold_per_item[..., None] - logits
    noise_required_to_win /= noise_std
    # p is the probability of being above the threshold for each (item, expert)
    # if the random noise (with its std) was re-sampled again.
    p = 1.0 - jax.scipy.stats.norm.cdf(noise_required_to_win)
    # We compute the average such probability for each expert over examples.
    p_mean = jnp.mean(p, axis=0)
    # Compute p_mean's coefficient of variation squared.
    return (jnp.std(p_mean) / jnp.mean(p_mean)) ** 2

  gates_logits = loss_parameters.moe_out.router_out.output
  noise_std = (1.0 / loss_parameters.num_experts) * loss_parameters.std_scale
  logits_noise = noise_std * jax.random.normal(
      key=loss_parameters.key, shape=gates_logits.shape
  )
  gates_logits_noisy = gates_logits + logits_noise
  load_term = jnp.mean(
      jax.vmap(
          functools.partial(
              calculate_load_loss,
              num_selected_experts=loss_parameters.num_selected_experts,
              noise_std=noise_std,
          )
      )(gates_logits, gates_logits_noisy)
  )
  weighted_load_loss = loss_parameters.load_weight * load_term

  statistics = [
      types.MoELossStatistic(
          name_id=types.NAME_TO_ID['LoadTerm'],  # pytype: disable=wrong-arg-types  # jnp-type
          value=load_term,
      ),
      types.MoELossStatistic(
          name_id=types.NAME_TO_ID['LoadLoss'],  # pytype: disable=wrong-arg-types  # jnp-type
          value=weighted_load_loss,
      ),
      types.MoELossStatistic(
          name_id=types.NAME_TO_ID['LoadWeight'],
          value=loss_parameters.load_weight,
      ),
  ]
  return types.MoELossReturn(value=weighted_load_loss, statistics=statistics)


@gin.configurable
def aux_loss(
    loss_parameters: types.MoELossParameters, loss_fns: Any = ()
) -> Iterable[types.MoELossReturn]:
  return [loss_fn(loss_parameters) for loss_fn in loss_fns]

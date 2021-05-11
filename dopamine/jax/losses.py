# coding=utf-8
# Copyright 2021 The Dopamine Authors.
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
"""Various losses used by the Dopamine JAX agents."""
from flax import linen as nn
import jax.numpy as jnp


def huber_loss(targets: jnp.array,
               predictions: jnp.array,
               delta: float = 1.0) -> jnp.ndarray:
  """Implementation of the Huber loss with threshold delta.

  Let `x = |targets - predictions|`, the Huber loss is defined as:
  `0.5 * x^2` if `x <= delta`
  `0.5 * delta^2 + delta * (x - delta)` otherwise.

  Args:
    targets: Target values.
    predictions: Prediction values.
    delta: Threshold.

  Returns:
    Huber loss.
  """
  x = jnp.abs(targets - predictions)
  return jnp.where(x <= delta,
                   0.5 * x**2,
                   0.5 * delta**2 + delta * (x - delta))


def mse_loss(targets: jnp.array, predictions: jnp.array) -> jnp.ndarray:
  """Implementation of the mean squared error loss."""
  return jnp.power((targets - predictions), 2)


def softmax_cross_entropy_loss_with_logits(labels: jnp.array,
                                           logits: jnp.array) -> jnp.ndarray:
  """Implementation of the softmax cross entropy loss."""
  return -jnp.sum(labels * nn.log_softmax(logits))

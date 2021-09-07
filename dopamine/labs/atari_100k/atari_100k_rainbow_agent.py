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
"""Atari 100k rainbow agent with support for data augmentation."""

import functools
from absl import logging
from dopamine.jax.agents.full_rainbow import full_rainbow_agent
import gin
import jax
import jax.numpy as jnp
import tensorflow as tf


############################ Data Augmentation ############################
def _random_crop(key, img, cropped_shape):
  """Random crop an image."""
  _, width, height = cropped_shape[:-1]
  key_x, key_y = jax.random.split(key, 2)
  x = jax.random.randint(key_x, shape=(), minval=0, maxval=img.shape[1] - width)
  y = jax.random.randint(
      key_y, shape=(), minval=0, maxval=img.shape[2] - height)
  return img[:, x:x + width, y:y + height]


def _intensity_aug(key, x, scale=0.05):
  """Follows the code in Schwarzer et al. (2020) for intensity augmentation."""
  r = jax.random.normal(key, shape=(x.shape[0], 1, 1, 1))
  noise = 1.0 + (scale * jnp.clip(r, -2.0, 2.0))
  return x * noise


def drq_image_augmentation(key, obs, img_pad=4):
  """Padding and cropping for DrQ (Kostrikov et. al, 2020)."""
  paddings = [(0, 0), (img_pad, img_pad), (img_pad, img_pad), (0, 0)]
  cropped_shape = obs.shape
  # The reference uses ReplicationPad2d in pytorch, but it is not available
  # in Jax. Use 'edge' instead.
  obs = jnp.pad(obs, paddings, 'edge')
  key1, key2 = jax.random.split(key, num=2)
  aug_obs = _intensity_aug(key1, _random_crop(key2, obs, cropped_shape))
  return aug_obs


def preprocess_inputs_with_augmentation(x, data_augmentation=False, rng=None):
  """Input normalization and if specified, data augmentation."""
  out = x.astype(jnp.float32) / 255.
  if data_augmentation:
    if rng is None:
      raise ValueError('Pass rng when using data augmentation')
    out = drq_image_augmentation(rng, out)
  return out


@gin.configurable
class Atari100kRainbowAgent(full_rainbow_agent.JaxFullRainbowAgent):
  """A compact implementation of agents for Atari 100k."""

  def __init__(self,
               num_actions,
               data_augmentation=False,
               summary_writer=None,
               seed=None):
    """Creates the Rainbow-based agent for the Atari 100k benchmark.

    On Atari 100k, an agent is evaluated after 100k environment steps, which
    corresponds to 2-3 hours of game play, for training.
    Args:
      num_actions: int, number of actions the agent can take at any state.
      data_augmentation: bool, whether to use data augmentation.
      summary_writer: SummaryWriter object, for outputting training statistics.
      seed: int, a seed for Jax RNG and initialization.
    """
    super().__init__(
        num_actions=num_actions,
        preprocess_fn=preprocess_inputs_with_augmentation,
        summary_writer=summary_writer,
        seed=seed)
    logging.info('\t data_augmentation: %s', data_augmentation)
    self._data_augmentation = data_augmentation
    logging.info('\t data_augmentation: %s', data_augmentation)
    # Preprocessing during training and evaluation can be possibly different,
    # for example, when using data augmentation during training.
    self.train_preprocess_fn = functools.partial(
        preprocess_inputs_with_augmentation,
        data_augmentation=data_augmentation)

  def _training_step_update(self):
    """Gradient update during every training step."""

    self._sample_from_replay_buffer()
    self._rng, rng1, rng2 = jax.random.split(self._rng, num=3)
    states = self.train_preprocess_fn(self.replay_elements['state'], rng=rng1)
    next_states = self.train_preprocess_fn(
        self.replay_elements['next_state'], rng=rng2)

    if self._replay_scheme == 'prioritized':
      probs = self.replay_elements['sampling_probabilities']
      # Weight the loss by the inverse priorities.
      loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
      loss_weights /= jnp.max(loss_weights)
    else:
      # Uniform weights if not using prioritized replay.
      loss_weights = jnp.ones(states.shape[0])

    (self.optimizer_state, self.online_params,
     loss, mean_loss, self._rng) = full_rainbow_agent.train(
         self.network_def, self.online_params, self.target_network_params,
         self.optimizer, self.optimizer_state, states,
         self.replay_elements['action'], next_states,
         self.replay_elements['reward'], self.replay_elements['terminal'],
         loss_weights, self._support, self.cumulative_gamma, self._double_dqn,
         self._distributional, self._rng)

    if self._replay_scheme == 'prioritized':
      self._replay.set_priority(self.replay_elements['indices'],
                                jnp.sqrt(loss + 1e-10))

    if self.summary_writer is not None:
      summary = tf.compat.v1.Summary(value=[
          tf.compat.v1.Summary.Value(
              tag='CrossEntropyLoss', simple_value=mean_loss)
      ])
      self.summary_writer.add_summary(summary, self.training_steps)


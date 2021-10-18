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
"""Tests for dopamine.labs.sac_from_pixels.continuous_networks."""

from absl.testing import absltest
from dopamine.labs.sac_from_pixels import continuous_networks
import jax
from jax import numpy as jnp


class SACEncoderNetworkTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.input_shape = (84, 84, 3)
    self.example_input = jnp.full(self.input_shape, 128, dtype=jnp.uint8)

  def test_network_outputs_correct_shapes(self):
    network_def = continuous_networks.SACEncoderNetwork()
    params = network_def.init(self.key, self.example_input)

    output = network_def.apply(params, self.example_input)

    self.assertIsInstance(output, continuous_networks.SACEncoderOutputs)
    with self.subTest('critic'):
      self.assertEqual(output.critic_z.shape, (50,))
    with self.subTest('actor'):
      self.assertEqual(output.actor_z.shape, (50,))

  def test_network_correctly_handles_stacked_rgb_frames(self):
    input_shape = (84, 84, 3, 4)  # 4 stacked RGB frames.
    example_input = jnp.zeros(input_shape, dtype=jnp.uint8)
    network_def = continuous_networks.SACEncoderNetwork()

    # Creating params with the example input is the first test.
    # If the input isn't handled correctly, this will throw an exception
    # and fail.
    params = network_def.init(self.key, example_input)

    # The first conv has a 3x3 filter, and 32 outputs. The input dimension
    # should be 3 filters x 4 stacks = 12.
    self.assertEqual(params['params']['Conv_0']['kernel'].shape, (3, 3, 12, 32))

  def test_actor_cant_update_conv_weights(self):
    network_def = continuous_networks.SACEncoderNetwork()
    params = network_def.init(self.key, self.example_input)

    def loss_fn(params):
      output = network_def.apply(params, self.example_input)
      return jnp.mean(output.actor_z)

    grad = jax.grad(loss_fn)(params)

    # This test only matters if the gradients are non-zero for the dense layer.
    # The actor's Dense layer is Dense_1.
    self.assertNotEqual(
        jnp.sum(jnp.abs(grad['params']['Dense_1']['kernel'])), 0.0)

    # It suffices to check that the final conv gradients are 0.
    self.assertEqual(jnp.sum(jnp.abs(grad['params']['Conv_3']['kernel'])), 0.0)
    self.assertEqual(jnp.sum(jnp.abs(grad['params']['Conv_3']['bias'])), 0.0)

  def test_critic_can_update_conv_weights(self):
    network_def = continuous_networks.SACEncoderNetwork()
    params = network_def.init(self.key, self.example_input)

    def loss_fn(params):
      output = network_def.apply(params, self.example_input)
      return jnp.mean(output.critic_z)

    grad = jax.grad(loss_fn)(params)

    # This test only matters if the gradients are non-zero for the dense layer.
    # The critic's Dense layer is Dense_0.
    self.assertNotEqual(
        jnp.sum(jnp.abs(grad['params']['Dense_0']['kernel'])), 0.0)

    # It suffices to check the final conv gradients are not 0.
    self.assertNotEqual(
        jnp.sum(jnp.abs(grad['params']['Conv_3']['kernel'])), 0.0)
    self.assertNotEqual(jnp.sum(jnp.abs(grad['params']['Conv_3']['bias'])), 0.0)


if __name__ == '__main__':
  absltest.main()

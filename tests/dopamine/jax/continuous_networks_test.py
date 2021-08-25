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
"""Tests for networks."""

from absl.testing import absltest
from dopamine.jax import continuous_networks
import jax
from jax import numpy as jnp


class SacCriticNetworkTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.example_state = jnp.zeros((8,), dtype=jnp.float32)
    self.example_action = jnp.zeros((3,), dtype=jnp.float32)
    self.action_shape = (4,)

  def test_sac_critic_network_outputs_single_value(self):
    network_def = continuous_networks.SACCriticNetwork()
    params = network_def.init(self.key, self.example_state, self.example_action)

    output = network_def.apply(params, self.example_state, self.example_action)

    self.assertEqual(output.shape, (1,))

  def test_network_has_specified_number_of_layers(self):
    num_layers = 5
    network_def = continuous_networks.SACCriticNetwork(num_layers=num_layers)
    params = network_def.init(self.key, self.example_state, self.example_action)

    # Checks which dense layers are found in the param dict.
    layer_names = {x for x in params['params'] if x.startswith('Dense_')}

    # n hidden layers + 1 output layer.
    self.assertLen(layer_names, num_layers + 1)


class SacNetworkTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.action_shape = (3,)
    self.num_layers = 3

    self.example_action = jnp.zeros(self.action_shape, dtype=jnp.float32)
    self.example_state = jnp.zeros((8,), dtype=jnp.float32)

    self.network_def = continuous_networks.SACNetwork(
        self.action_shape, self.num_layers)
    self.params = self.network_def.init(self.key, self.example_state, self.key)

  def assert_actor_shapes_are_correct(
      self, actor_output: continuous_networks.SacActorOutput):
    with self.subTest(name='mean_action'):
      self.assertEqual(actor_output.mean_action.shape, self.action_shape)
    with self.subTest(name='sampled_action'):
      self.assertEqual(actor_output.sampled_action.shape, self.action_shape)
    with self.subTest(name='log_probability'):
      self.assertEqual(actor_output.log_probability.shape, ())

  def assert_critic_shapes_are_correct(
      self, critic_output: continuous_networks.SacCriticOutput):
    with self.subTest(name='q_value1'):
      self.assertEqual(critic_output.q_value1.shape, (1,))
    with self.subTest(name='q_value2'):
      self.assertEqual(critic_output.q_value2.shape, (1,))

  def test_sac_network_call_outputs_correct_shaped_values(self):
    output = self.network_def.apply(self.params, self.example_state, self.key)

    self.assertIsInstance(output, continuous_networks.SacOutput)
    self.assert_actor_shapes_are_correct(output.actor)
    self.assert_critic_shapes_are_correct(output.critic)

  def test_sac_network_actor_outputs_correct_shaped_values(self):
    output = self.network_def.apply(
        self.params,
        self.example_state,
        self.key,
        method=self.network_def.actor)

    self.assertIsInstance(output, continuous_networks.SacActorOutput)
    self.assert_actor_shapes_are_correct(output)

  def test_sac_network_critic_outputs_correct_shaped_values(self):
    output = self.network_def.apply(
        self.params,
        self.example_state,
        self.example_action,
        method=self.network_def.critic)

    self.assertIsInstance(output, continuous_networks.SacCriticOutput)
    self.assert_critic_shapes_are_correct(output)

  def test_network_has_specified_number_of_layers(self):
    # Checks which actor layers are found in the param dict.
    actor_layer_names = {x for x in self.params['params']
                         if x.startswith('_actor_layers_')}
    with self.subTest('actor'):
      self.assertLen(actor_layer_names, self.num_layers)
      self.assertIn('_actor_final_layer', self.params['params'])

    critic1_layer_names = {x for x in self.params['params']['_critic1']
                           if x.startswith('Dense_')}
    critic2_layer_names = {x for x in self.params['params']['_critic2']
                           if x.startswith('Dense_')}
    with self.subTest('critic1'):
      # n hidden layers + 1 output layer.
      self.assertLen(critic1_layer_names, self.num_layers + 1)
    with self.subTest('critic2'):
      # n hidden layers + 1 output layer.
      self.assertLen(critic2_layer_names, self.num_layers + 1)


if __name__ == '__main__':
  absltest.main()

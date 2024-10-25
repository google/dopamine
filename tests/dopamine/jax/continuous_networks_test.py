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
from absl.testing import parameterized
from dopamine.jax import continuous_networks
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np


class ActorNetworkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.example_state = jnp.zeros((8,), dtype=jnp.float32)
    self.action_shape = (4,)

  def test_actor_network_outputs_correct_shaped_values(self):
    network_def = continuous_networks.ActorNetwork(self.action_shape)
    params = network_def.init(self.key, self.example_state, self.key)

    mode, sampled_action, action_probability = network_def.apply(
        params, self.example_state, self.key
    )

    self.assertEqual(mode.shape, self.action_shape)
    self.assertEqual(sampled_action.shape, self.action_shape)
    self.assertEqual(action_probability.shape, ())

  def test_network_has_specified_number_of_layers(self):
    num_layers = 5
    network_def = continuous_networks.ActorNetwork(
        self.action_shape, num_layers
    )
    params = network_def.init(self.key, self.example_state, self.key)

    # Checks which dense layers are found in the param dict.
    layer_names = {x for x in params['params'] if x.startswith('Dense_')}

    self.assertLen(layer_names, num_layers + 1)

  @parameterized.named_parameters(
      ('relu', nn.relu), ('sigmoid', nn.sigmoid), ('tanh', nn.tanh)
  )
  def test_network_activation_initializer(self, activation):
    network = continuous_networks.ActorNetwork(
        self.action_shape, activation=activation
    )
    self.assertEqual(network.activation, activation)

  @parameterized.named_parameters(
      ('glorot_uniform', jax.nn.initializers.glorot_uniform()),
      ('glorot_normal', jax.nn.initializers.glorot_normal()),
      ('uniform', jax.nn.initializers.uniform()),
  )
  def test_network_kernel_initializer(self, kernel_initializer):
    network = continuous_networks.ActorNetwork(
        self.action_shape, kernel_initializer=kernel_initializer
    )
    self.assertEqual(network.kernel_initializer, kernel_initializer)


class CriticNetworkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.example_state = jnp.zeros((8,), dtype=jnp.float32)
    self.example_action = jnp.zeros((3,), dtype=jnp.float32)
    self.action_shape = (4,)

  def test_critic_network_outputs_single_value(self):
    network_def = continuous_networks.CriticNetwork()
    params = network_def.init(self.key, self.example_state, self.example_action)

    output = network_def.apply(params, self.example_state, self.example_action)

    self.assertEqual(output.shape, (1,))

  def test_network_has_specified_number_of_layers(self):
    num_layers = 5
    network_def = continuous_networks.CriticNetwork(num_layers=num_layers)
    params = network_def.init(self.key, self.example_state, self.example_action)

    # Checks which dense layers are found in the param dict.
    layer_names = {x for x in params['params'] if x.startswith('Dense_')}

    # n hidden layers + 1 output layer.
    self.assertLen(layer_names, num_layers + 1)

  @parameterized.named_parameters(
      ('relu', nn.relu), ('sigmoid', nn.sigmoid), ('tanh', nn.tanh)
  )
  def test_network_activation_initializer(self, activation):
    network = continuous_networks.CriticNetwork(activation=activation)
    self.assertEqual(network.activation, activation)

  @parameterized.named_parameters(
      ('glorot_uniform', jax.nn.initializers.glorot_uniform()),
      ('glorot_normal', jax.nn.initializers.glorot_normal()),
      ('uniform', jax.nn.initializers.uniform()),
  )
  def test_network_kernel_initializer(self, kernel_initializer):
    network = continuous_networks.CriticNetwork(
        kernel_initializer=kernel_initializer
    )
    self.assertEqual(network.kernel_initializer, kernel_initializer)


class ActorCriticNetworkTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.action_shape = (3,)
    self.num_layers = 3

    self.example_action = jnp.zeros(self.action_shape, dtype=jnp.float32)
    self.example_state = jnp.zeros((8,), dtype=jnp.float32)

    self.network_def = continuous_networks.ActorCriticNetwork(
        self.action_shape, self.num_layers
    )
    self.params = self.network_def.init(self.key, self.example_state, self.key)

  def assert_actor_shapes_are_correct(
      self, actor_output: continuous_networks.ActorOutput
  ):
    with self.subTest(name='mean_action'):
      self.assertEqual(actor_output.mean_action.shape, self.action_shape)
    with self.subTest(name='sampled_action'):
      self.assertEqual(actor_output.sampled_action.shape, self.action_shape)
    with self.subTest(name='log_probability'):
      self.assertEqual(actor_output.log_probability.shape, ())

  def assert_critic_shapes_are_correct(
      self, critic_output: continuous_networks.CriticOutput
  ):
    with self.subTest(name='q_value1'):
      self.assertEqual(critic_output.q_value1.shape, (1,))
    with self.subTest(name='q_value2'):
      self.assertEqual(critic_output.q_value2.shape, (1,))

  def test_actor_critic_network_call_outputs_correct_shaped_values(self):
    output = self.network_def.apply(self.params, self.example_state, self.key)

    self.assertIsInstance(output, continuous_networks.ActorCriticOutput)
    self.assert_actor_shapes_are_correct(output.actor)
    self.assert_critic_shapes_are_correct(output.critic)

  def test_actor_critic_network_actor_outputs_correct_shaped_values(self):
    output = self.network_def.apply(
        self.params, self.example_state, self.key, method=self.network_def.actor
    )

    self.assertIsInstance(output, continuous_networks.ActorOutput)
    self.assert_actor_shapes_are_correct(output)

  def test_actor_critic_network_critic_outputs_correct_shaped_values(self):
    output = self.network_def.apply(
        self.params,
        self.example_state,
        self.example_action,
        method=self.network_def.critic,
    )

    self.assertIsInstance(output, continuous_networks.CriticOutput)
    self.assert_critic_shapes_are_correct(output)

  def test_network_has_specified_number_of_layers(self):
    actor_layer_names = {
        x for x in self.params['params']['_actor'] if x.startswith('Dense_')
    }
    with self.subTest('actor'):
      # n hidden layers + 1 output layer.
      self.assertLen(actor_layer_names, self.num_layers + 1)

    critic1_layer_names = {
        x for x in self.params['params']['_critic1'] if x.startswith('Dense_')
    }
    critic2_layer_names = {
        x for x in self.params['params']['_critic2'] if x.startswith('Dense_')
    }
    with self.subTest('critic1'):
      # n hidden layers + 1 output layer.
      self.assertLen(critic1_layer_names, self.num_layers + 1)
    with self.subTest('critic2'):
      # n hidden layers + 1 output layer.
      self.assertLen(critic2_layer_names, self.num_layers + 1)


class PPOActorNetworkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.example_state = jnp.zeros((8,), dtype=jnp.float32)
    self.action_shape = (4,)

  def test_ppo_actor_network_outputs_correct_shaped_values(self):
    network_def = continuous_networks.PPOActorNetwork(self.action_shape)
    params = network_def.init(self.key, self.example_state, self.key)

    sampled_action, action_probability, entropy = network_def.apply(
        params, self.example_state, self.key
    )

    self.assertEqual(sampled_action.shape, self.action_shape)
    self.assertEqual(action_probability.shape, ())
    self.assertEqual(entropy.shape, ())

  def test_ppo_network_has_specified_number_of_layers(self):
    num_layers = 5
    network_def = continuous_networks.PPOActorNetwork(
        self.action_shape, num_layers
    )
    params = network_def.init(self.key, self.example_state, self.key)

    # Checks which dense layers are found in the param dict.
    layer_names = {x for x in params['params'] if x.startswith('Dense_')}

    # n hidden layers + 2 output layers.
    self.assertLen(layer_names, num_layers + 2)

  @parameterized.named_parameters(
      ('relu', nn.relu), ('sigmoid', nn.sigmoid), ('tanh', nn.tanh)
  )
  def test_ppo_network_activation_initializer(self, activation):
    network = continuous_networks.PPOActorNetwork(
        self.action_shape, activation=activation
    )
    self.assertEqual(network.activation, activation)

  def test_scale_diag_set_to_zero(self):
    num_layers = 4
    hidden_units = 256
    network = continuous_networks.PPOActorNetwork(
        self.action_shape, num_layers, hidden_units
    )
    params = network.init(self.key, self.example_state, self.key)
    np.testing.assert_array_equal(
        params['params'][f'Dense_{num_layers+1}']['kernel'],
        jnp.zeros((hidden_units,) + self.action_shape),
    )


class PPOCriticNetworkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.example_state = jnp.zeros((8,), dtype=jnp.float32)
    self.action_shape = (4,)

  def test_ppo_critic_network_outputs_single_value(self):
    network_def = continuous_networks.PPOCriticNetwork()
    params = network_def.init(self.key, self.example_state)

    output = network_def.apply(params, self.example_state)

    self.assertEqual(output.shape, (1,))

  def test_ppo_network_has_specified_number_of_layers(self):
    num_layers = 5
    network_def = continuous_networks.PPOCriticNetwork(num_layers=num_layers)
    params = network_def.init(self.key, self.example_state)

    # Checks which dense layers are found in the param dict.
    layer_names = {x for x in params['params'] if x.startswith('Dense_')}

    # n hidden layers + 1 output layer.
    self.assertLen(layer_names, num_layers + 1)

  @parameterized.named_parameters(
      ('relu', nn.relu), ('sigmoid', nn.sigmoid), ('tanh', nn.tanh)
  )
  def test_ppo_network_activation_initializer(self, activation):
    network = continuous_networks.PPOCriticNetwork(activation=activation)
    self.assertEqual(network.activation, activation)


class PPOActorCriticNetworkTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.key = jax.random.PRNGKey(0)
    self.action_shape = (3,)
    self.num_layers = 3

    self.example_state = jnp.zeros((8,), dtype=jnp.float32)

    self.network_def = continuous_networks.PPOActorCriticNetwork(
        self.action_shape, self.num_layers
    )
    self.params = self.network_def.init(self.key, self.example_state, self.key)

  def assert_ppo_actor_shapes_are_correct(
      self, actor_output: continuous_networks.PPOActorOutput
  ):
    with self.subTest(name='sampled_action'):
      self.assertEqual(actor_output.sampled_action.shape, self.action_shape)
    with self.subTest(name='log_probability'):
      self.assertEqual(actor_output.log_probability.shape, ())
    with self.subTest(name='entropy'):
      self.assertEqual(actor_output.entropy.shape, ())

  def assert_ppo_critic_shapes_are_correct(
      self, critic_output: continuous_networks.PPOCriticOutput
  ):
    with self.subTest(name='q_value'):
      self.assertEqual(critic_output.q_value.shape, (1,))

  def test_ppo_actor_critic_network_call_outputs_correct_shaped_values(self):
    output = self.network_def.apply(self.params, self.example_state, self.key)

    self.assertIsInstance(output, continuous_networks.PPOActorCriticOutput)
    self.assert_ppo_actor_shapes_are_correct(output.actor)
    self.assert_ppo_critic_shapes_are_correct(output.critic)

  def test_ppo_actor_critic_network_actor_outputs_correct_shaped_values(self):
    output = self.network_def.apply(
        self.params, self.example_state, self.key, method=self.network_def.actor
    )

    self.assertIsInstance(output, continuous_networks.PPOActorOutput)
    self.assert_ppo_actor_shapes_are_correct(output)

  def test_ppo_actor_critic_network_critic_outputs_correct_shaped_values(self):
    output = self.network_def.apply(
        self.params,
        self.example_state,
        method=self.network_def.critic,
    )

    self.assertIsInstance(output, continuous_networks.PPOCriticOutput)
    self.assert_ppo_critic_shapes_are_correct(output)

  def test_network_has_specified_number_of_layers(self):
    actor_layer_names = {
        x for x in self.params['params']['_actor'] if x.startswith('Dense_')
    }
    with self.subTest('actor'):
      # n hidden layers + 2 output layers.
      self.assertLen(actor_layer_names, self.num_layers + 2)

    critic_layer_names = {
        x for x in self.params['params']['_critic'] if x.startswith('Dense_')
    }
    with self.subTest('critic'):
      # n hidden layers + 1 output layer.
      self.assertLen(critic_layer_names, self.num_layers + 1)


class ContinousNetworksHelperTest(parameterized.TestCase):

  @parameterized.parameters(('relu', nn.relu), ('tanh', nn.tanh))
  def test_create_activation(self, name, activation):
    created_activation = continuous_networks.create_activation(name)
    self.assertEqual(created_activation, activation)

  def test_create_activation_raises_error(self):
    with self.assertRaises(ValueError):
      continuous_networks.create_activation('not_an_activation')


if __name__ == '__main__':
  absltest.main()

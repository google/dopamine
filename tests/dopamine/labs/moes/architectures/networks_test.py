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
"""Tests for moes.networks."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
from dopamine.labs.moes.architectures import networks
from dopamine.labs.moes.architectures import types
import jax
import numpy as np


class NetworksTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._pixel_state = np.ones((84, 84, 4))
    self._vector_state = np.ones((84,))
    self._num_actions = 13
    self._num_experts = 5
    self._expert_hidden_size = 128
    self._rng = jax.random.PRNGKey(42)

  @parameterized.named_parameters(
      dict(testcase_name='MaintainTokenSize', maintain_token_size=True),
      dict(testcase_name='NoMaintainTokenSize', maintain_token_size=False),
  )
  def testExpertModel(self, maintain_token_size):
    net = networks.ExpertModel(
        expert_hidden_size=self._expert_hidden_size,
        rng_key=jax.random.PRNGKey(0),
        maintain_token_size=maintain_token_size,
    )
    params = net.init(self._rng, x=self._vector_state)
    expected_output_size = (
        self._vector_state.shape[0]
        if maintain_token_size
        else self._expert_hidden_size,
    )

    net_output = net.apply(params, self._vector_state)
    np.testing.assert_equal(net_output[0].shape, expected_output_size)

  def testImpalaMoEWithInvalidMoEType(self):
    net = networks.ImpalaMoE(
        num_actions=4, inputs_preprocessed=True, moe_type='FOOBAR'
    )
    with self.assertRaises(AssertionError):
      _ = net.init(self._rng, x=self._pixel_state, key=self._rng)

  def testImpalaMoEWithInvalidRoutingType(self):
    net = networks.ImpalaMoE(
        num_actions=4, inputs_preprocessed=True, routing_type='FOOBAR'
    )
    with self.assertRaises(AssertionError):
      _ = net.init(self._rng, x=self._pixel_state, key=self._rng)

  def testImpalaMoEBaseline(self):
    net = networks.ImpalaMoE(
        num_actions=self._num_actions,
        inputs_preprocessed=True,
        moe_type='BASELINE',
    )
    params = net.init(self._rng, x=self._pixel_state, key=self._rng)
    net_out = net.apply(params, self._pixel_state, key=self._rng)
    np.testing.assert_equal(net_out.q_values.shape, (self._num_actions,))
    self.assertIsInstance(net_out, types.BaselineNetworkReturn)

  def _create_network_and_apply(self, network_class, moe_type, state,
                                support=None):
    net = network_class(
        num_actions=self._num_actions,
        inputs_preprocessed=True,
        num_experts=self._num_experts,
        moe_type=moe_type,
    )
    if support is None:
      params = net.init(self._rng, x=state, key=self._rng)
      return net.apply(params, state, key=self._rng)

    params = net.init(self._rng, x=state, key=self._rng, support=support)
    return net.apply(params, state, support, key=self._rng)

  def _create_impala_network_and_apply(self, moe_type, routing_type):
    return self._create_network_and_apply(
        network_class=functools.partial(
            networks.ImpalaMoE, routing_type=routing_type, patch_size=(4, 4)
        ),
        moe_type=moe_type,
        state=self._pixel_state,
    )

  def _test_network_outputs(
      self,
      net_out,
      expected_moe_output_shape,
      expected_router_out_shape,
      expected_expert_weights_shape,
      expected_top_expert_shape,
  ):
    router_out = net_out.moe_out.router_out
    np.testing.assert_equal(net_out.q_values.shape, (self._num_actions,))
    self.assertIsInstance(net_out, types.MoENetworkReturn)
    np.testing.assert_equal(
        net_out.moe_out.values.shape, expected_moe_output_shape
    )
    np.testing.assert_equal(router_out.output.shape, expected_router_out_shape)
    np.testing.assert_equal(
        router_out.probabilities.shape, expected_router_out_shape
    )
    np.testing.assert_equal(
        router_out.top_expert_weights.shape, expected_expert_weights_shape
    )
    np.testing.assert_equal(
        router_out.top_experts.shape, expected_top_expert_shape
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='PerPixel',
          routing_type='PER_PIXEL',
          expected_moe_output_shape=(121, 32),
      ),
      dict(
          testcase_name='PerFeatureMap',
          routing_type='PER_FEATUREMAP',
          expected_moe_output_shape=(32, 121),
      ),
      dict(
          testcase_name='PerSample',
          routing_type='PER_SAMPLE',
          expected_moe_output_shape=(1, 32 * 121),
      ),
      dict(
          testcase_name='PerPatch',
          routing_type='PER_PATCH',
          expected_moe_output_shape=(4, 32),
      ),
  )  # Using (4, 4) patches.
  def testImpalaMoE(self, routing_type, expected_moe_output_shape):
    net_out = self._create_impala_network_and_apply('MOE', routing_type)
    self._test_network_outputs(
        net_out,
        expected_moe_output_shape,
        expected_router_out_shape=(
            expected_moe_output_shape[0],
            self._num_experts,
        ),
        expected_expert_weights_shape=(expected_moe_output_shape[0], 1),
        expected_top_expert_shape=(expected_moe_output_shape[0], 1),
    )


  @parameterized.named_parameters(
      dict(
          testcase_name='PerPixel',
          routing_type='PER_PIXEL',
          expected_moe_output_shape=(121, 32),
      ),
      dict(
          testcase_name='PerFeatureMap',
          routing_type='PER_FEATUREMAP',
          expected_moe_output_shape=(32, 121),
      ),
      dict(
          testcase_name='PerPatch',
          routing_type='PER_PATCH',
          expected_moe_output_shape=(4, 32),
      ),
  )  # Using (4, 4) patches.
  def testImpalaSoftMoE(self, routing_type, expected_moe_output_shape):
    net_out = self._create_impala_network_and_apply('SOFTMOE', routing_type)
    self._test_network_outputs(
        net_out,
        expected_moe_output_shape,
        expected_router_out_shape=(
            expected_moe_output_shape[0],
            self._num_experts,
        ),
        expected_expert_weights_shape=(1,),
        expected_top_expert_shape=(expected_moe_output_shape[0], 1),
    )

  def testNatureDQNMoEWithInvalidMoEType(self):
    net = networks.NatureDQNMoE(
        num_actions=4, inputs_preprocessed=True, moe_type='FOOBAR'
    )
    with self.assertRaises(AssertionError):
      _ = net.init(self._rng, x=self._pixel_state, key=self._rng)

  def testNatureDQNMoEWithInvalidRoutingType(self):
    net = networks.NatureDQNMoE(
        num_actions=4, inputs_preprocessed=True, routing_type='FOOBAR'
    )
    with self.assertRaises(AssertionError):
      _ = net.init(self._rng, x=self._pixel_state, key=self._rng)

  def testNatureDQNMoEBaseline(self):
    net = networks.NatureDQNMoE(
        num_actions=self._num_actions,
        inputs_preprocessed=True,
        moe_type='BASELINE',
    )
    params = net.init(self._rng, x=self._pixel_state, key=self._rng)
    net_out = net.apply(params, self._pixel_state, key=self._rng)
    np.testing.assert_equal(net_out.q_values.shape, (self._num_actions,))
    self.assertIsInstance(net_out, types.BaselineNetworkReturn)

  def _create_nature_network_and_apply(self, moe_type, routing_type):
    return self._create_network_and_apply(
        network_class=functools.partial(
            networks.NatureDQNMoE, routing_type=routing_type, patch_size=(4, 4)
        ),
        moe_type=moe_type,
        state=self._pixel_state,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='PerPixel',
          routing_type='PER_PIXEL',
          expected_moe_output_shape=(121, 64),
      ),
      dict(
          testcase_name='PerFeatureMap',
          routing_type='PER_FEATUREMAP',
          expected_moe_output_shape=(64, 121),
      ),
      dict(
          testcase_name='PerSample',
          routing_type='PER_SAMPLE',
          expected_moe_output_shape=(1, 64 * 121),
      ),
      dict(
          testcase_name='PerPatch',
          routing_type='PER_PATCH',
          expected_moe_output_shape=(4, 64),
      ),
  )  # Using (4, 4) patches.
  def testNatureDQNMoE(self, routing_type, expected_moe_output_shape):
    net_out = self._create_nature_network_and_apply('MOE', routing_type)
    self._test_network_outputs(
        net_out,
        expected_moe_output_shape,
        expected_router_out_shape=(
            expected_moe_output_shape[0],
            self._num_experts,
        ),
        expected_expert_weights_shape=(expected_moe_output_shape[0], 1),
        expected_top_expert_shape=(expected_moe_output_shape[0], 1),
    )


  @parameterized.named_parameters(
      dict(
          testcase_name='PerPixel',
          routing_type='PER_PIXEL',
          expected_moe_output_shape=(121, 64),
      ),
      dict(
          testcase_name='PerFeatureMap',
          routing_type='PER_FEATUREMAP',
          expected_moe_output_shape=(64, 121),
      ),
      dict(
          testcase_name='PerPatch',
          routing_type='PER_PATCH',
          expected_moe_output_shape=(4, 64),
      ),
  )  # Using (4, 4) patches.
  def testNatureDQNSoftMoE(self, routing_type, expected_moe_output_shape):
    net_out = self._create_nature_network_and_apply('SOFTMOE', routing_type)
    self._test_network_outputs(
        net_out,
        expected_moe_output_shape,
        expected_router_out_shape=(expected_moe_output_shape[0], 5),
        expected_expert_weights_shape=(1,),
        expected_top_expert_shape=(expected_moe_output_shape[0], 1),
    )

  def _create_full_rainbow_network_and_apply(self, moe_type, routing_type):
    return self._create_network_and_apply(
        network_class=functools.partial(
            networks.FullRainbowMoENetwork, routing_type=routing_type,
            patch_size=(4, 4), num_atoms=51,
            distributional=False,  # To avoid dealing with support.
            encoder_type='CNN',
        ),
        moe_type=moe_type,
        state=self._pixel_state,
        support=np.ones(51)
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='PerPixel',
          routing_type='PER_PIXEL',
          expected_moe_output_shape=(121, 64),
      ),
      dict(
          testcase_name='PerFeatureMap',
          routing_type='PER_FEATUREMAP',
          expected_moe_output_shape=(64, 121),
      ),
      dict(
          testcase_name='PerSample',
          routing_type='PER_SAMPLE',
          expected_moe_output_shape=(1, 64 * 121),
      ),
      dict(
          testcase_name='PerPatch',
          routing_type='PER_PATCH',
          expected_moe_output_shape=(4, 64),
      ),
  )  # Using (4, 4) patches.
  def testFullRainbowMoE(self, routing_type, expected_moe_output_shape):
    net_out = self._create_full_rainbow_network_and_apply('MOE', routing_type)
    self._test_network_outputs(
        net_out,
        expected_moe_output_shape,
        expected_router_out_shape=(
            expected_moe_output_shape[0],
            self._num_experts,
        ),
        expected_expert_weights_shape=(expected_moe_output_shape[0], 1),
        expected_top_expert_shape=(expected_moe_output_shape[0], 1),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='PerPixel',
          routing_type='PER_PIXEL',
          expected_moe_output_shape=(121, 64),
      ),
      dict(
          testcase_name='PerFeatureMap',
          routing_type='PER_FEATUREMAP',
          expected_moe_output_shape=(64, 121),
      ),
      dict(
          testcase_name='PerPatch',
          routing_type='PER_PATCH',
          expected_moe_output_shape=(4, 64),
      ),
  )  # Using (4, 4) patches.
  def testFullRainbowSoftMoE(self, routing_type, expected_moe_output_shape):
    net_out = self._create_full_rainbow_network_and_apply(
        'SOFTMOE', routing_type
    )
    self._test_network_outputs(
        net_out,
        expected_moe_output_shape,
        expected_router_out_shape=(expected_moe_output_shape[0], 5),
        expected_expert_weights_shape=(1,),
        expected_top_expert_shape=(expected_moe_output_shape[0], 1),
    )



if __name__ == '__main__':
  absltest.main()

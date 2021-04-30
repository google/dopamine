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
"""Tests for dopamine.jax.losses."""

from typing import Optional, Union

from absl.testing import absltest
from absl.testing import parameterized
from dopamine.jax import losses
import numpy as onp


class LossesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='BelowDelta1d',
           target=1.0, prediction=0.0, delta=1.0,
           expected=onp.array(0.5)),
      dict(testcase_name='AboveDelta1d',
           target=1.0, prediction=0.0, delta=0.5,
           expected=onp.array(0.375)),
      dict(testcase_name='MixedArraysDefaultDelta',
           target=onp.ones(5), prediction=onp.array([0., 1., 2., 3., 4.]),
           delta=None,
           expected=onp.array([0.5, 0., 0.5, 1.5, 2.5])),
      dict(testcase_name='MixedArraysSetDelta',
           target=onp.ones(5), prediction=onp.array([0., 1., 2., 3., 4.]),
           delta=2.0,
           expected=onp.array([0.5, 0., 0.5, 2.0, 4.0])))
  def testHuberLoss(self,
                    target: Union[float, onp.array],
                    prediction: Union[float, onp.array],
                    delta: Optional[float],
                    expected: Union[float, onp.array]):
    if delta is None:
      actual = losses.huber_loss(target, prediction)
    else:
      actual = losses.huber_loss(target, prediction, delta=delta)
    onp.testing.assert_equal(actual, expected)

  @parameterized.named_parameters(
      dict(testcase_name='1DParameters',
           target=2.0, prediction=0.0, expected=onp.array(4.0)),
      dict(testcase_name='ArrayParameters',
           target=onp.ones(5), prediction=onp.array([0., 1., 2., 3., 4.]),
           expected=onp.array([1.0, 0.0, 1.0, 4.0, 9.0])))
  def testMSELoss(self,
                  target: Union[float, onp.array],
                  prediction: Union[float, onp.array],
                  expected: Union[float, onp.array]):
    actual = losses.mse_loss(target, prediction)
    onp.testing.assert_equal(actual, expected)


if __name__ == '__main__':
  absltest.main()

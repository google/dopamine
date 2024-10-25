# coding=utf-8
# Copyright 2024 The Dopamine Authors.
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
"""Serialization test."""

from typing import Union

from absl.testing import absltest
from absl.testing import parameterized
from dopamine.jax import serialization
import numpy as np


class SerializationTest(parameterized.TestCase):

  @parameterized.parameters(
      (np.array([1, 2, 3], order='F'),),
      (np.array([1, 2, 3], order='C'),),
      (np.array([1, 2, 3]),),
      (np.zeros((4, 4, 4), order='F'),),
      (np.zeros((4, 4, 4), order='C'),),
      (np.zeros((4, 4, 4)),),
      # Structured array
      (
          np.array(
              [('A', 1, 1.0), ('B', 2, 2.0)],
              dtype=[('string', 'U1'), ('int', 'i4'), ('float', 'f4')],
          ),
      ),
      (np.bool_(True),),
      (np.int_(1),),
      (np.float64(1.0),),
  )
  def testEncodeNumpy(self, array: Union[np.ndarray, np.bool_, np.number]):
    encoded = serialization.encode(array)
    self.assertIn('dtype', encoded)
    self.assertIsInstance(encoded['dtype'], str)
    self.assertIn('shape', encoded)
    self.assertIsInstance(encoded['shape'], tuple)
    self.assertIn('data', encoded)
    self.assertIsInstance(encoded['data'], bytes)

    decoded = serialization.decode(encoded)
    np.testing.assert_array_equal(array, decoded, strict=True)

  @parameterized.parameters((1,), (1234567891011121314151617181920,))
  def testEncodeLongIntegers(self, integer: int):
    encoded = serialization.encode(integer)

    if integer.bit_length() > 32:
      self.assertIsInstance(encoded, dict)
      self.assertIn('integer', encoded)
      self.assertIsInstance(encoded['integer'], str)
    else:
      assert isinstance(encoded, int)
      assert encoded == integer

    decoded = serialization.decode(encoded)
    assert decoded == integer


if __name__ == '__main__':
  absltest.main()

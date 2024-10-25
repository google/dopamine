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
"""Testing samplers."""

from absl.testing import absltest
from absl.testing import parameterized
from dopamine.jax.replay_memory import samplers
import numpy as np


class UniformSamplingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.sampler = samplers.UniformSamplingDistribution(0)

  def test_update_does_not_raise_and_logs(self):
    with self.assertLogs(level="INFO") as logs:
      self.sampler.update(0, None, [], arg_x=1, arg_y=2)
    self.assertLen(logs.output, 1)
    self.assertContainsSubsequence(logs.output[0], "is a no-op")

  def test_additional_kwargs_to_add_logs(self):
    with self.assertLogs(level="INFO") as logs:
      self.sampler.add(1, dummy_kwargs="dummy")
    self.assertLen(logs.output, 1)
    self.assertContainsSubsequence(logs.output[0], "dummy_kwargs")

  def test_sample_when_empty(self):
    with self.assertRaises(ValueError):
      self.sampler.sample(1)

  def test_removal_of_invalid_key_raises(self):
    with self.assertRaises(ValueError):
      self.sampler.remove(1)

  @parameterized.parameters((0,), (-1,))
  def test_invalid_sample_size_raises(self, size: int):
    with self.assertRaises(ValueError):
      self.sampler.sample(size)

  def test_add_and_remove(self):
    self.sampler.add(1)
    self.sampler.remove(1)

  def test_sample(self):
    self.sampler.add(1)
    sample = self.sampler.sample(1)
    self.assertIsInstance(sample, samplers.SampleMetadata)
    self.assertLen(sample.keys, 1)
    self.assertEqual(sample.keys[0], 1)

  def test_serializes(self):
    sampler = samplers.UniformSamplingDistribution(0)
    sampler.add(1)
    state_dict = sampler.to_state_dict()
    self.assertIn("key_by_index", state_dict)
    self.assertIn("index_by_key", state_dict)
    self.assertIn("rng_state", state_dict)

    sampler = samplers.UniformSamplingDistribution(0)
    sampler.from_state_dict(state_dict)
    self.assertEqual(sampler.sample(1).keys, 1)

  def test_clear_sampler(self):
    self.sampler.add(1)
    self.assertNotEmpty(self.sampler._key_by_index)
    self.assertNotEmpty(self.sampler._index_by_key)
    self.sampler.clear()
    self.assertEmpty(self.sampler._key_by_index)
    self.assertEmpty(self.sampler._index_by_key)


class PrioritizedSamplingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.sampler = samplers.PrioritizedSamplingDistribution(
        seed=0, max_capacity=10
    )

  def test_priorities_can_be_updated(self):
    keys = [0, 1, 2, 3, 4]
    priorities = [1.0, 1.0, 1.0, 1.0, 1.0]

    for key, priority in zip(keys, priorities):
      self.sampler.add(key, priority=priority)

    generator_state = self.sampler._rng.bit_generator.state
    self.sampler._rng.bit_generator.state = generator_state
    sample1 = self.sampler.sample(5)

    priorities = [0.1, 0.9, 5.0, 0.25, 1.0]
    for key, priority in zip(keys, priorities):
      self.sampler.add(key, priority=priority)

    self.sampler._rng.bit_generator.state = generator_state
    sample2 = self.sampler.sample(5)

    with self.assertRaises(AssertionError):
      np.testing.assert_array_equal(sample1.keys, sample2.keys)

  def test_priorities_can_be_removed(self):
    keys = [0, 1, 2, 3, 4]
    priorities = [1.0, 2.0, 3.0, 4.0, 5.0]

    for key, priority in zip(keys, priorities):
      self.sampler.add(key, priority=priority)

    self.sampler.sample(5)
    for key in keys:
      self.sampler.remove(key)

    with self.assertRaises(ValueError):
      self.sampler.sample(5)

  def test_zero_priorities_is_uniform_sampling(self):
    for index in range(3):
      self.sampler.add(index, priority=0.0)

    samples = self.sampler.sample(3)
    np.testing.assert_allclose(samples.probabilities, 1 / 3)

  def test_positive_priorities_computes_probabilities(self):
    for index in range(3):
      self.sampler.add(index, priority=1.0)

    samples = self.sampler.sample(3)
    np.testing.assert_allclose(samples.probabilities, 1 / 3)

  @parameterized.parameters((0,), (1,), (2,))
  def test_removal_wont_sample_removed_index(self, index_to_remove: int):
    for index in range(3):
      self.sampler.add(index, priority=1.0)

    self.sampler.remove(index_to_remove)
    samples = self.sampler.sample(1000)
    self.assertNoCommonElements(samples.keys, [index_to_remove])

  def test_clear_sampler(self):
    for index in range(3):
      self.sampler.add(index, priority=1.0)
    self.assertNotEmpty(self.sampler._key_by_index)
    self.assertNotEmpty(self.sampler._index_by_key)
    self.sampler.clear()
    self.assertEmpty(self.sampler._key_by_index)
    self.assertEmpty(self.sampler._index_by_key)
    self.assertEqual(self.sampler._sum_tree.root, 0.0)
    for index in range(3):
      self.assertEqual(self.sampler._sum_tree.get(index), 0.0)


class SequentialSamplingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.sampler = samplers.SequentialSamplingDistribution(seed=0)

  def test_sample_when_empty(self):
    with self.assertRaises(ValueError):
      self.sampler.sample(1)

  def test_sample_when_not_empty(self):
    keys = [1, 2, 3]
    for key in keys:
      self.sampler.add(key)
    sample = self.sampler.sample(3)
    self.assertIsInstance(sample, samplers.SampleMetadata)
    self.assertLen(sample.keys, 3)
    for sample_key, key in zip(sample.keys, keys):
      self.assertEqual(sample_key, key)

  def test_order_is_sequential_after_add(self):
    keys = [1, 2, 4]
    for key in keys:
      self.sampler.add(key)
    self.sampler.add(3)
    sample = self.sampler.sample(4)
    self.assertIsInstance(sample, samplers.SampleMetadata)
    self.assertLen(sample.keys, 4)
    keys = [1, 2, 3, 4]
    for sample_key, key in zip(sample.keys, keys):
      self.assertEqual(sample_key, key)

  def test_order_is_sequential_after_remove(self):
    keys = [1, 2, 3, 4, 5]
    for key in keys:
      self.sampler.add(key)
    self.sampler.remove(2)
    sample = self.sampler.sample(4)
    self.assertIsInstance(sample, samplers.SampleMetadata)
    self.assertLen(sample.keys, 4)
    keys = [1, 3, 4, 5]
    for sample_key, key in zip(sample.keys, keys):
      self.assertEqual(sample_key, key)

  def test_order_with_unsorted_add(self):
    keys = [4, 1, 3, 5, 2]
    for key in keys:
      self.sampler.add(key)
    sample = self.sampler.sample(5)
    self.assertIsInstance(sample, samplers.SampleMetadata)
    self.assertLen(sample.keys, 5)
    keys = [1, 2, 3, 4, 5]
    for sample_key, key in zip(sample.keys, keys):
      self.assertEqual(sample_key, key)

  def test_clear_sampler(self):
    for i in range(3):
      self.sampler.add(i)
    self.assertNotEmpty(self.sampler._key_by_index)
    self.assertNotEmpty(self.sampler._index_by_key)
    self.sampler.clear()
    self.assertEmpty(self.sampler._key_by_index)
    self.assertEmpty(self.sampler._index_by_key)

  def test_unsorted_add_with_sort_samples_false(self):
    sampler = samplers.SequentialSamplingDistribution(
        seed=0, sort_samples=False
    )
    keys = [4, 1, 3, 5, 2]
    for key in keys:
      sampler.add(key)
    sample = sampler.sample(5)
    self.assertIsInstance(sample, samplers.SampleMetadata)
    self.assertLen(sample.keys, 5)
    for sample_key, key in zip(sample.keys, keys):
      self.assertEqual(sample_key, key)


if __name__ == "__main__":
  absltest.main()

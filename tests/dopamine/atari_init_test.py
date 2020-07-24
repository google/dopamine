# coding=utf-8
# Copyright 2018 The Dopamine Authors.
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
"""A simple test for validating that the Atari env initializes."""

import datetime
import os
import shutil



from absl import flags
from dopamine.discrete_domains import train
import tensorflow as tf


FLAGS = flags.FLAGS


class AtariInitTest(tf.test.TestCase):

  def setUp(self):
    super(AtariInitTest, self).setUp()
    FLAGS.base_dir = os.path.join(
        '/tmp/dopamine_tests',
        datetime.datetime.utcnow().strftime('run_%Y_%m_%d_%H_%M_%S'))
    FLAGS.gin_files = ['dopamine/agents/dqn/configs/dqn.gin']
    # `num_iterations` set to zero to prevent runner execution.
    FLAGS.gin_bindings = [
        'Runner.num_iterations=0',
        'WrappedReplayBuffer.replay_capacity = 100'  # To prevent OOM.
    ]
    FLAGS.alsologtostderr = True

  def test_atari_init(self):
    """Tests that a DQN agent is initialized."""
    train.main([])
    shutil.rmtree(FLAGS.base_dir)


if __name__ == '__main__':
  tf.compat.v1.disable_v2_behavior()
  tf.test.main()

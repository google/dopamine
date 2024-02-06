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
r"""Sample file to generate visualizations.

To run, point FLAGS.restore_checkpoint to the TensorFlow checkpoint of a
trained agent. As an example, you can download to `/tmp/checkpoints` the files
linked below:
  # pylint: disable=line-too-long
  * https://storage.cloud.google.com/download-dopamine-rl/colab/samples/rainbow/SpaceInvaders_v4/checkpoints/tf_ckpt-199.data-00000-of-00001
  * https://storage.cloud.google.com/download-dopamine-rl/colab/samples/rainbow/SpaceInvaders_v4/checkpoints/tf_ckpt-199.index
  * https://storage.cloud.google.com/download-dopamine-rl/colab/samples/rainbow/SpaceInvaders_v4/checkpoints/tf_ckpt-199.meta
  # pylint: enable=line-too-long

You can then run the binary with:

```
python example_viz.py \
        --agent='rainbow' \
        --game='SpaceInvaders' \
        --num_steps=1000 \
        --root_dir='/tmp/dopamine' \
        --restore_checkpoint=/tmp/checkpoints/colab_samples_rainbow_SpaceInvaders_v4_checkpoints_tf_ckpt-199
```

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from absl import app
from absl import flags
from dopamine.utils import example_viz_lib

flags.DEFINE_string('agent', 'dqn', 'Agent to visualize.')
flags.DEFINE_string('game', 'Breakout', 'Game to visualize.')
flags.DEFINE_string('root_dir', '/tmp/dopamine/', 'Root directory.')
flags.DEFINE_string(
    'restore_checkpoint', None, 'Path to checkpoint to restore for visualizing.'
)
flags.DEFINE_integer('num_steps', 2000, 'Number of steps to run.')
flags.DEFINE_boolean(
    'use_legacy_checkpoint',
    False,
    'Set to true if loading from a legacy (pre-Keras) checkpoint.',
)

FLAGS = flags.FLAGS


def main(_):
  example_viz_lib.run(
      agent=FLAGS.agent,
      game=FLAGS.game,
      num_steps=FLAGS.num_steps,
      root_dir=FLAGS.root_dir,
      restore_ckpt=FLAGS.restore_checkpoint,
      use_legacy_checkpoint=FLAGS.use_legacy_checkpoint,
  )


if __name__ == '__main__':
  app.run(main)

# coding=utf-8
"""Sample file to generate visualizations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import app
from absl import flags
from dopamine.utils import example_viz_lib

flags.DEFINE_string('agent', 'dqn', 'Agent to visualize.')
flags.DEFINE_string('game', 'Breakout', 'Game to visualize.')
flags.DEFINE_string('root_dir', '/tmp/dopamine/', 'Root directory.')
flags.DEFINE_integer('num_steps', 2000, 'Number of steps to run.')

FLAGS = flags.FLAGS


def main(_):
  example_viz_lib.run(agent=FLAGS.agent,
                      game=FLAGS.game,
                      num_steps=FLAGS.num_steps,
                      root_dir=FLAGS.root_dir)

if __name__ == '__main__':
  app.run(main)

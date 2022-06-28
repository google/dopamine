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
r"""Entry point for Atari 100k experiments.

"""

import functools
import json
import os

from absl import app
from absl import flags
from absl import logging
from dopamine.discrete_domains import run_experiment
from dopamine.discrete_domains import train as base_train
from dopamine.labs.atari_100k import atari_100k_rainbow_agent
from dopamine.labs.atari_100k import eval_run_experiment
import numpy as np
import tensorflow as tf


FLAGS = flags.FLAGS
AGENTS = ['DER', 'DrQ', 'OTRainbow', 'DrQ_eps']

# flags are defined when importing run_xm_preprocessing
flags.DEFINE_enum('agent', 'DER', AGENTS, 'Name of the agent.')
flags.DEFINE_integer('run_number', 1, 'Run number.')
flags.DEFINE_boolean('max_episode_eval', True,
                     'Whether to use `MaxEpisodeEvalRunner` or not.')


def create_agent(sess,  # pylint: disable=unused-argument
                 environment,
                 seed,
                 summary_writer=None):
  """Helper function for creating full rainbow-based Atari 100k agent."""
  return atari_100k_rainbow_agent.Atari100kRainbowAgent(
      num_actions=environment.action_space.n,
      seed=seed,
      summary_writer=summary_writer)


def set_random_seed(seed):
  """Set random seed for reproducibility."""
  logging.info('Setting random seed: %d', seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  tf.random.set_seed(seed)
  np.random.seed(seed)


def main(unused_argv):
  """Main method.

  Args:
    unused_argv: Arguments (unused).
  """
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.enable_v2_behavior()

  base_dir = FLAGS.base_dir
  gin_files, gin_bindings = FLAGS.gin_files, FLAGS.gin_bindings
  run_experiment.load_gin_configs(gin_files, gin_bindings)
  # Set the Jax agent seed using the run number
  create_agent_fn = functools.partial(create_agent, seed=FLAGS.run_number)
  if FLAGS.max_episode_eval:
    runner_fn = eval_run_experiment.MaxEpisodeEvalRunner
    logging.info('Using MaxEpisodeEvalRunner for evaluation.')
    runner = runner_fn(base_dir, create_agent_fn)
  else:
    runner = run_experiment.Runner(base_dir, create_agent_fn)
  runner.run_experiment()


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)

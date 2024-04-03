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
r"""Entry point for Atari 100k experiments with MoEs."""

import functools
import json
import os

from absl import app
from absl import flags
from absl import logging
from dopamine.discrete_domains import run_experiment
from dopamine.discrete_domains import train as base_train
from dopamine.labs.atari_100k import atari_100k_runner
from dopamine.labs.moes.agents import rainbow_100k_moe_agent
import numpy as np
import tensorflow as tf

.learning.deepmind.xmanager2.client.google as xm  # pylint: disable=unused-import


FLAGS = flags.FLAGS
AGENTS = ['moe_der']

# flags are defined when importing run_xm_preprocessing
flags.DEFINE_enum('agent', 'moe_der', AGENTS, 'Name of the agent.')
flags.DEFINE_integer('run_number', 1, 'Run number.')
flags.DEFINE_boolean(
    'max_episode_eval', True, 'Whether to use `MaxEpisodeEvalRunner` or not.'
)
flags.DEFINE_boolean(
    'legacy_runner',
    False,  # TODO(psc): Make it work with better runner.
    (
        'Whether to use the legacy MaxEpisodeEvalRunner.'
        ' This runner does not run parallel evaluation environments and may be'
        ' easier to understand, but will be noticeably slower. It also does not'
        ' guarantee that a precise number of training steps will be collected,'
        ' which clashes with the technical definition of Atari 100k.'
    ),
)


def create_agent(
    sess,  # pylint: disable=unused-argument
    environment,
    seed,
    agent_name: str,
    summary_writer=None,
):
  """Helper function for creating full rainbow-based Atari 100k agent."""

  del agent_name
  return rainbow_100k_moe_agent.Atari100kRainbowMoEAgent(
      num_actions=environment.action_space.n,
      seed=seed,
      summary_writer=summary_writer,
  )


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
  xm_xid = None if 'xm_xid' not in FLAGS else FLAGS.xm_xid
  xm_wid = None if 'xm_wid' not in FLAGS else FLAGS.xm_wid
  xm_parameters = None if 'xm_parameters' not in FLAGS else FLAGS.xm_parameters
  if xm_parameters:
    xm_params = json.loads(xm_parameters)
    if 'run_number' in xm_params:
      FLAGS.run_number = xm_params['run_number']
      logging.info('Run number set to: %d', xm_params['run_number'])
  # Add code for setting random seed using the run_number
  set_random_seed(FLAGS.run_number)
  base_dir, gin_files, gin_bindings = base_train.run_xm_preprocessing(
      xm_xid,
      xm_wid,
      xm_parameters,
      FLAGS.base_dir,
      FLAGS.custom_base_dir_from_hparams,
      FLAGS.gin_files,
      FLAGS.gin_bindings,
  )
  run_experiment.load_gin_configs(gin_files, gin_bindings)
  # Set the Jax agent seed using the run number
  create_agent_fn = functools.partial(
      create_agent, seed=FLAGS.run_number, agent_name=FLAGS.agent
  )
  if FLAGS.legacy_runner:
    runner = run_experiment.Runner(base_dir, create_agent_fn)
  else:
    runner = atari_100k_runner.DataEfficientAtariRunner(
        base_dir, create_agent_fn
    )
  runner.run_experiment()


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)

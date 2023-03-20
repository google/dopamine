# coding=utf-8
# Copyright 2023 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""The entry point for running experiments.

"""

import functools
import json
import os

from absl import app
from absl import flags
from absl import logging
from dopamine.discrete_domains import run_experiment as base_run_experiment
from dopamine.discrete_domains import train as base_train
from dopamine.labs.offline_rl.jax import offline_dqn_agent
from dopamine.labs.offline_rl.jax import offline_dr3_agent
from dopamine.labs.offline_rl.jax import offline_dr3_with_validation_agent
from dopamine.labs.offline_rl.jax import offline_rainbow_agent
from dopamine.labs.offline_rl.jax import return_conditioned_bc_agent
from dopamine.labs.offline_rl.jax import run_experiment
from jax.config import config as jax_config

AGENTS = [
    'jax_dqn', 'jax_dr3', 'jax_rainbow', 'jax_return_conditioned_bc',
    'jax_dr3_with_validation']

# flags are defined when importing run_xm_preprocessing
flags.DEFINE_enum('agent_name', 'jax_dqn', AGENTS, 'Name of the agent.')
flags.DEFINE_string('replay_dir', None, 'Directory from which to load the '
                    'replay data')
flags.DEFINE_string('replay_dir_suffix', 'replay_logs', 'Data is to be read '
                    'from "replay_dir/.../{replay_dir_suffix}"')
flags.DEFINE_bool('use_tfds', True, 'Use tfds data')
flags.DEFINE_boolean('disable_jit', False, 'Whether to use jit or not.')


FLAGS = flags.FLAGS


def create_offline_agent(sess,
                         environment,
                         agent_name,
                         replay_data_dir,
                         summary_writer=None):
  """Creates an online agent.

  Args:
    sess: A `tf.Session` object for running associated ops. This argument is
      ignored for JAX agents.
    environment: An Atari 2600 environment.
    agent_name: Name of the agent to be created.
    replay_data_dir: Directory from which to load the fixed replay buffers.
    summary_writer: A Tensorflow summary writer to pass to the agent for
      in-agent training statistics in Tensorboard.

  Returns:
    An agent with metrics.
  """
  if agent_name == 'jax_dqn':
    agent = offline_dqn_agent.OfflineJaxDQNAgent
  elif agent_name == 'jax_dr3':
    agent = offline_dr3_agent.OfflineJaxDR3Agent
  elif agent_name == 'jax_rainbow':
    agent = offline_rainbow_agent.OfflineJaxRainbowAgent
  elif agent_name == 'jax_return_conditioned_bc':
    agent = return_conditioned_bc_agent.JaxReturnConditionedBCAgent
  elif agent_name == 'jax_dr3_with_validation':
    agent = offline_dr3_with_validation_agent.OfflineJaxDR3WithValidationAgent
  else:
    raise ValueError('{} is not a valid agent name'.format(FLAGS.agent_name))

  if agent_name.startswith('jax'):  # JAX agent
    return agent(
        num_actions=environment.action_space.n,
        replay_data_dir=replay_data_dir,
        summary_writer=summary_writer,
        use_tfds=FLAGS.use_tfds)
  else:
    return agent(
        sess,
        num_actions=environment.action_space.n,
        replay_data_dir=replay_data_dir,
        summary_writer=summary_writer)


def create_replay_dir(xm_parameters):
  """Creates the replay data directory from xm_parameters."""
  replay_dir = FLAGS.replay_dir
  if xm_parameters:
    xm_params = json.loads(xm_parameters)
    problem_name, run_number = '', ''
    for param, value in xm_params.items():
      if param.endswith('game_name'):
        problem_name = value
      elif param.endswith('run_number'):
        run_number = str(value)
    replay_dir = os.path.join(replay_dir, problem_name, run_number)
  return os.path.join(replay_dir, FLAGS.replay_dir_suffix)


def main(unused_argv):
  """Main method.

  Args:
    unused_argv: Arguments (unused).
  """
  logging.set_verbosity(logging.INFO)
  if FLAGS.disable_jit:
    jax_config.update('jax_disable_jit', True)

  xm_xid = None if 'xm_xid' not in FLAGS else FLAGS.xm_xid
  xm_wid = None if 'xm_wid' not in FLAGS else FLAGS.xm_wid
  xm_parameters = (None
                   if 'xm_parameters' not in FLAGS else FLAGS.xm_parameters)
  base_dir, gin_files, gin_bindings = base_train.run_xm_preprocessing(
      xm_xid, xm_wid, xm_parameters, FLAGS.base_dir,
      FLAGS.custom_base_dir_from_hparams, FLAGS.gin_files, FLAGS.gin_bindings)
  create_agent = functools.partial(
      create_offline_agent,
      agent_name=FLAGS.agent_name,
      replay_data_dir=create_replay_dir(xm_parameters))
  base_run_experiment.load_gin_configs(gin_files, gin_bindings)
  runner = run_experiment.FixedReplayRunner(base_dir, create_agent)
  runner.run_experiment()


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)

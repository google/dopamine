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
"""Brax environments made compatible for Dopamine."""

import time
from typing import Any, Mapping, Optional, Tuple

from brax import envs
from dopamine.continuous_domains import run_experiment
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.agents.sac import sac_agent
from flax.metrics import tensorboard
import gin
import jax
import numpy as onp


class BraxEnv(object):
  """Wrapper class for Brax environments."""

  def __init__(self, env_name: str, seed: Optional[int] = None):
    self.env = envs.create(env_name=env_name)
    self.game_over = False
    seed = int(time.time() * 1e6) if seed is None else seed
    self._rng = jax.random.PRNGKey(seed)
    self._rng, rng = jax.random.split(self._rng)
    self._state = self.env.reset(rng=rng)

  @property
  def observation_space(self) -> onp.ndarray:
    return self._state.obs.shape

  @property
  def action_space(self) -> int:
    return self.env.action_size

  @property
  def reward_range(self):
    pass  # Unused

  @property
  def metadata(self):
    pass  # Unused

  def reset(self) -> onp.ndarray:
    self.game_over = False
    self._rng, rng = jax.random.split(self._rng)
    self._state = self.env.reset(rng=rng)
    return self._state.obs

  def step(self, action) -> Tuple[onp.ndarray, onp.ndarray, bool,
                                  Mapping[Any, Any]]:
    self._state = jax.jit(self.env.step)(self._state, action)
    self.game_over = self._state.done
    return (onp.array(self._state.obs),
            onp.array(self._state.reward),
            self._state.done,
            self._state.info)


@gin.configurable
def create_brax_environment(env_name, seed=None) -> BraxEnv:
  """Helper function for creating a Brax environment."""
  return BraxEnv(env_name, seed=seed)


@gin.configurable
def create_brax_agent(
    environment: BraxEnv,
    agent_name: str = 'sac_brax',
    summary_writer: Optional[tensorboard.SummaryWriter] = None
) -> dqn_agent.JaxDQNAgent:
  """Creates an agent for Brax."""
  assert agent_name is not None
  if agent_name == 'sac_brax':
    return sac_agent.SACAgent(
        action_shape=(environment.action_space,),
        action_limits=(-1 * onp.ones(environment.action_space),
                       onp.ones(environment.action_space)),
        observation_shape=environment.observation_space,
        action_dtype=onp.float32,
        observation_dtype=onp.float64,
        summary_writer=summary_writer)
  else:
    raise ValueError(f'Unknown agent: {agent_name}')


@gin.configurable
def create_brax_runner(
    base_dir: str,
    schedule: str = 'continuous_train_and_eval'
) -> run_experiment.ContinuousRunner:
  """Creates a Brax experiment Runner."""
  assert base_dir is not None
  # Continuously runs training and evaluation until max num_iterations is hit.
  if schedule == 'continuous_train_and_eval':
    return run_experiment.ContinuousRunner(base_dir, create_brax_agent)
  # Continuously runs training until max num_iterations is hit.
  elif schedule == 'continuous_train':
    return run_experiment.ContinuousTrainRunner(base_dir, create_brax_agent)
  else:
    raise ValueError('Unknown schedule: {}'.format(schedule))

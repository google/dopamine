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
"""Compact implementation of an offline BC + Rainbow agent in JAX."""

import enum
import functools

from absl import logging
from dopamine.google.experiments.two_hot import losses as classy_transforms
from dopamine.jax import losses
from dopamine.jax.agents.full_rainbow import full_rainbow_agent
from dopamine.jax.agents.rainbow import rainbow_agent
from dopamine.labs.offline_rl import fixed_replay
from dopamine.labs.offline_rl.jax import networks  # pylint: disable=unused-import
from dopamine.labs.offline_rl.jax import offline_dr3_agent
from dopamine.labs.offline_rl.rlu_tfds import tfds_replay
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tensorflow as tf


classification_loss_fn = jax.vmap(losses.softmax_cross_entropy_loss_with_logits)
mse_loss_fn = jax.vmap(losses.mse_loss)


class ClassyLoss(enum.StrEnum):
  HL_GAUSS = 'hl_gauss'
  TWO_HOT = 'two_hot'
  BINARY_CROSSENTROPY = 'binary_crossentropy'
  SCALAR = 'scalar'  # Uses softmax to compute scalar q-values


class TargetType(enum.StrEnum):
  MAXQ = 'q_learning'
  SARSA = 'sarsa'
  MC = 'monte_carlo'  # Uses Return to Go values.


@functools.partial(jax.vmap, in_axes=(None, 0, None))
def get_network_outputs(model, states, rng):
  outputs = model(states, key=rng)
  return (
      jnp.squeeze(outputs.logits),
      jnp.squeeze(outputs.q_values),
      jnp.squeeze(outputs.representation),
  )


@functools.partial(
    jax.vmap, in_axes=(None, None, 0, 0, 0, 0, None, None, None, None)
)
def target_output(
    model,
    target_network,
    next_states,
    next_actions,
    rewards,
    terminals,
    cumulative_gamma,
    double_dqn,
    target_type,
    rng,
):
  """Builds the C51 target distribution or DQN target Q-values."""

  is_terminal_multiplier = 1.0 - terminals.astype(jnp.float32)
  # Incorporate terminal state to discount factor.
  gamma_with_terminal = cumulative_gamma * is_terminal_multiplier

  target_network_dist = target_network(next_states, key=rng)
  if double_dqn:
    # Use the current network for the action selection
    next_state_target_outputs = model(next_states, key=rng)
  else:
    next_state_target_outputs = target_network_dist
  # Action selection using Q-values for next-state
  q_values = jnp.squeeze(next_state_target_outputs.q_values)
  if target_type == TargetType.SARSA:
    backup_actions = next_actions
  elif target_type == TargetType.MAXQ:
    backup_actions = jnp.argmax(q_values)
  else:
    raise NotImplementedError(f'Unknown target type: {target_type}')

  # Compute the target Q-value
  next_q_values = jnp.squeeze(target_network_dist.q_values)
  replay_next_qt = next_q_values[backup_actions]
  target = rewards + gamma_with_terminal * replay_next_qt
  return jax.lax.stop_gradient(target)


@functools.partial(
    jax.jit,
    static_argnames=(
        'network_def',
        'optimizer',
        'double_dqn',
        'cumulative_gamma',
        'target_type',
        'bc_coefficient',
        'td_coefficient',
    ),
)
def train(
    network_def,
    online_params,
    target_params,
    optimizer,
    optimizer_state,
    states,
    actions,
    next_states,
    next_actions,
    rewards,
    terminals,
    monte_carlo_returns,
    support,
    cumulative_gamma,
    double_dqn,
    rng,
    target_type,
    bc_coefficient=0.0,
    td_coefficient=1.0,
):
  """Run a training step."""

  # Split the current rng into 2 for updating the rng after this call
  rng, rng1, rng2 = jax.random.split(rng, num=3)

  def q_online(state, key):
    return network_def.apply(online_params, state, key=key, support=support)

  def q_target(state, key):
    return network_def.apply(target_params, state, key=key, support=support)

  def loss_fn(params, target):
    """Computes the distributional loss for C51 or huber loss for DQN."""

    def q_func(state, key):
      return network_def.apply(params, state, key=key, support=support)

    logits, q_values, curr_repr = get_network_outputs(q_func, states, rng)
    _, _, next_repr = get_network_outputs(q_func, next_states, rng)
    replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
    if network_def.transform is not None:
      target_probs = jax.vmap(network_def.transform.transform_to_probs)(target)
      # Fetch the logits for its selected action. We use vmap to perform this
      # indexing across the batch.
      chosen_action_logits = jax.vmap(lambda x, y: x[y])(logits, actions)
      mean_td_loss = jnp.mean(
          classification_loss_fn(target_probs, chosen_action_logits)
      )
    else:
      mean_td_loss = jnp.mean(mse_loss_fn(target, replay_chosen_q))

    mc_loss = jnp.mean(mse_loss_fn(monte_carlo_returns, replay_chosen_q))
    dr3_loss = offline_dr3_agent.compute_dr3_loss(curr_repr, next_repr)
    bc_loss = jnp.mean(
        jax.scipy.special.logsumexp(q_values, axis=-1) - replay_chosen_q
    )
    mean_loss = td_coefficient * mean_td_loss + bc_coefficient * bc_loss
    avg_q_value = jnp.mean(replay_chosen_q)
    return mean_loss, (mean_td_loss, bc_loss, dr3_loss, mc_loss, avg_q_value)

  if target_type == TargetType.MC:
    target = monte_carlo_returns
  else:
    # Compute scalar target Q-values.
    target = target_output(
        q_online,
        q_target,
        next_states,
        next_actions,
        rewards,
        terminals,
        cumulative_gamma,
        double_dqn,
        target_type,
        rng1,
    )

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  # Get the unweighted loss without taking its mean for updating priorities.
  # outputs[1] correspond to the per-example TD loss.
  (loss, outputs), grad = grad_fn(online_params, target)
  updates, optimizer_state = optimizer.update(
      grad, optimizer_state, params=online_params
  )
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, loss, outputs, rng2


@gin.configurable
class OfflineClassyCQLAgent(full_rainbow_agent.JaxFullRainbowAgent):
  """Offline ClassyDQN agent with BC regularization (akin to CQL)."""

  def __init__(
      self,
      num_actions,
      replay_data_dir,
      network=networks.ParameterizedRainbowNetwork,
      td_coefficient=1.0,
      bc_coefficient=0.0,
      summary_writer=None,
      add_return_to_go=False,
      target_type: TargetType = TargetType.MAXQ,
      hl_loss_type: ClassyLoss = ClassyLoss.HL_GAUSS,
      hl_gauss_sigma_ratio: float = 0.75,
      replay_buffer_builder=None,
      use_tfds=True,
  ):
    """Initializes the agent and constructs the necessary components.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      replay_data_dir: str, log Directory from which to load the replay buffer.
      network: Q-network to use.
      td_coefficient: float, Coefficient for the DR3 regularizer.
      bc_coefficient: float, Coefficient for the CQL loss.
      summary_writer: SummaryWriter object for outputting training statistics.
      add_return_to_go: Whether to add return to go to the replay data.
      target_type: Whether to use SARSA, MC or Q-learning targets.
      hl_loss_type: Histogram loss function for classification.
      hl_gauss_sigma_ratio: std parameter for HL-gauss loss.
      replay_buffer_builder: Callable object that takes "self" as an argument
        and returns a replay buffer to use for training offline. If None, it
        will use the default FixedReplayBuffer.
      use_tfds: Whether to use tfds replay buffer.
    """

    add_return_to_go = add_return_to_go or (target_type == TargetType.MC)

    logging.info('Creating OfflineJaxRainbowAgent with the parameters:')
    logging.info('\t replay directory: %s', replay_data_dir)
    logging.info('\t TD coefficient: %s', td_coefficient)
    logging.info('\t CQL coefficient: %s', bc_coefficient)
    logging.info('\t Return To Go: %s', add_return_to_go)
    logging.info('\t TargetType: %s', target_type)
    logging.info('\t hl_loss_type: %s', hl_loss_type)

    assert target_type in [TargetType.MC, TargetType.MAXQ, TargetType.SARSA]

    self.replay_data_dir = replay_data_dir
    self._use_tfds = use_tfds
    self._td_coefficient = td_coefficient
    self._bc_coefficient = bc_coefficient
    self._add_return_to_go = add_return_to_go
    self._target_type = target_type

    if replay_buffer_builder is not None:
      self._build_replay_buffer = replay_buffer_builder

    vmax = float(gin.query_parameter('JaxFullRainbowAgent.vmax'))
    vmin = -vmax
    num_atoms = gin.query_parameter('JaxFullRainbowAgent.num_atoms')

    assert hl_loss_type in [
        ClassyLoss.HL_GAUSS,
        ClassyLoss.TWO_HOT,
        ClassyLoss.SCALAR,
    ]
    match hl_loss_type:
      case ClassyLoss.HL_GAUSS:
        self._transform = classy_transforms.GaussianHistogramLoss(
            num_bins=num_atoms,
            sigma_ratio=hl_gauss_sigma_ratio,
            min_value=vmin,
            max_value=vmax,
            widen_support=False,
            apply_symlog=False,
        )

      case ClassyLoss.TWO_HOT:
        self._transform = classy_transforms.TwoHotHistogramLoss(
            num_bins=num_atoms,
            min_value=vmin,
            max_value=vmax,
            apply_symlog=False,
        )
      case ClassyLoss.BINARY_CROSSENTROPY:
        self._transform = classy_transforms.BinaryCrossEntropyLoss(
            min_value=vmin,
            max_value=vmax,
            apply_symlog=False,
        )
      case ClassyLoss.SCALAR:
        self._transform = None

    super().__init__(
        num_actions,
        noisy=False,  # No need for noisy networks for offline RL.
        dueling=False,  # Set dueling networks also to be False by default.
        replay_scheme='uniform',  # Uniform replay is default for offline RL.
        network=functools.partial(
            networks.ParameterizedRainbowNetwork, transform=self._transform
        ),
        summary_writer=summary_writer,
    )

  def train_step(self):
    """Runs a single training step."""
    for _ in range(self._num_updates_per_train_step):
      self._training_step_update()

  def _training_step_update(self):
    """Runs a single training step."""
    self._sample_from_replay_buffer()
    states = self.preprocess_fn(self.replay_elements['state'])
    next_states = self.preprocess_fn(self.replay_elements['next_state'])
    if self._target_type == TargetType.MC:
      mc_returns = self.replay_elements['return_to_go']
    else:
      mc_returns = jnp.zeros_like(self.replay_elements['reward'])

    (
        self.optimizer_state,
        self.online_params,
        mean_loss,
        aux_info,
        self._rng,
    ) = train(
        self.network_def,
        self.online_params,
        self.target_network_params,
        self.optimizer,
        self.optimizer_state,
        states,
        self.replay_elements['action'],
        next_states,
        self.replay_elements['next_action'],
        self.replay_elements['reward'],
        self.replay_elements['terminal'],
        mc_returns,
        self._support,
        self.cumulative_gamma,
        self._double_dqn,
        self._rng,
        target_type=self._target_type,
        bc_coefficient=self._bc_coefficient,
        td_coefficient=self._td_coefficient,
    )

    td_loss, bc_loss, dr3_loss, mc_loss, avg_q_value = aux_info
    if (
        self.training_steps > 0
        and self.training_steps % self.summary_writing_frequency == 0
    ):
      if self.summary_writer is not None:
        with self.summary_writer.as_default():
          tf.summary.scalar(
              'Losses/Aggregate', mean_loss, step=self.training_steps
          )
          tf.summary.scalar('Losses/TD', td_loss, step=self.training_steps)
          tf.summary.scalar('Losses/BCLoss', bc_loss, step=self.training_steps)
          tf.summary.scalar(
              'Losses/DR3Loss', dr3_loss, step=self.training_steps
          )
          tf.summary.scalar('Losses/MCLoss', mc_loss, step=self.training_steps)
          tf.summary.scalar(
              'Losses/AvgQ', avg_q_value, step=self.training_steps
          )
        self.summary_writer.flush()
    if self._use_tfds:
      self.log_gradient_steps_per_epoch()

    if self.training_steps % self.target_update_period == 0:
      self._sync_weights()
    self.training_steps += 1

  def _build_replay_buffer(self):
    """Creates the fixed replay buffer used by the agent."""

    if not self._use_tfds:
      return fixed_replay.JaxFixedReplayBuffer(
          data_dir=self.replay_data_dir,
          observation_shape=self.observation_shape,
          stack_size=self.stack_size,
          update_horizon=self.update_horizon,
          gamma=self.gamma,
          observation_dtype=self.observation_dtype,
      )

    dataset_name = tfds_replay.get_atari_ds_name_from_replay(
        self.replay_data_dir
    )
    return tfds_replay.JaxFixedReplayBufferTFDS(
        replay_capacity=gin.query_parameter(
            'JaxFixedReplayBuffer.replay_capacity'
        ),
        batch_size=gin.query_parameter('JaxFixedReplayBuffer.batch_size'),
        dataset_name=dataset_name,
        stack_size=self.stack_size,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        return_to_go=self._add_return_to_go,
    )

  def log_gradient_steps_per_epoch(self):
    num_steps_per_epoch = self._replay.gradient_steps_per_epoch
    steps_per_epoch = self.training_steps / num_steps_per_epoch
    if self.summary_writer is not None:
      with self.summary_writer.as_default():
        tf.summary.scalar(
            'Info/EpochFractionSteps', steps_per_epoch, step=self.training_steps
        )

  def _sample_from_replay_buffer(self):
    if self._use_tfds:
      self.replay_elements = self._replay.sample_transition_batch()
    else:
      super()._sample_from_replay_buffer()

  def reload_data(self):
    # This doesn't do anything for tfds replay.
    self._replay.reload_data()

  def step(self, reward, observation):
    """Returns the agent's next action and update agent's state.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    self._record_observation(observation)
    self._rng, self.action = rainbow_agent.select_action(
        self.network_def,
        self.online_params,
        self.state,
        self._rng,
        self.num_actions,
        self.eval_mode,
        self.epsilon_eval,
        self.epsilon_train,
        self.epsilon_decay_period,
        self.training_steps,
        self.min_replay_history,
        self.epsilon_fn,
        self._support,
    )
    self.action = onp.asarray(self.action)
    return self.action

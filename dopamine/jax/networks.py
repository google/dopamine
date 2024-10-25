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
"""Various networks for Jax Dopamine agents."""

import functools
import itertools
import operator
import time
from typing import Optional, Sequence, Tuple, Union

from absl import logging
from dopamine.discrete_domains import atari_lib
from dopamine.jax import continuous_networks
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import numpy as onp
from tensorflow_probability.substrates import jax as tfp


gin.constant('jax_networks.CARTPOLE_OBSERVATION_DTYPE', jnp.float64)
gin.constant(
    'jax_networks.CARTPOLE_MIN_VALS',
    (-2.4, -5.0, -onp.pi / 12.0, -onp.pi * 2.0),
)
gin.constant(
    'jax_networks.CARTPOLE_MAX_VALS', (2.4, 5.0, onp.pi / 12.0, onp.pi * 2.0)
)
gin.constant('jax_networks.ACROBOT_OBSERVATION_DTYPE', jnp.float64)
gin.constant(
    'jax_networks.ACROBOT_MIN_VALS', (-1.0, -1.0, -1.0, -1.0, -5.0, -5.0)
)
gin.constant('jax_networks.ACROBOT_MAX_VALS', (1.0, 1.0, 1.0, 1.0, 5.0, 5.0))
gin.constant('jax_networks.LUNAR_OBSERVATION_DTYPE', jnp.float64)
gin.constant('jax_networks.MOUNTAINCAR_OBSERVATION_DTYPE', jnp.float64)
gin.constant('jax_networks.MOUNTAINCAR_MIN_VALS', (-1.2, -0.07))
gin.constant('jax_networks.MOUNTAINCAR_MAX_VALS', (0.6, 0.07))


def preprocess_atari_inputs(x):
  """Input normalization for Atari 2600 input frames."""
  return x.astype(jnp.float32) / 255.0


identity_preprocess_fn = lambda x: x


@gin.configurable
class Stack(nn.Module):
  """Stack of pooling and convolutional blocks with residual connections."""

  num_ch: int
  num_blocks: int
  use_max_pooling: bool = True

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    conv_out = nn.Conv(
        features=self.num_ch,
        kernel_size=(3, 3),
        strides=1,
        kernel_init=initializer,
        padding='SAME',
    )(x)
    if self.use_max_pooling:
      conv_out = nn.max_pool(
          conv_out, window_shape=(3, 3), padding='SAME', strides=(2, 2)
      )

    for _ in range(self.num_blocks):
      block_input = conv_out
      conv_out = nn.relu(conv_out)
      conv_out = nn.Conv(
          features=self.num_ch, kernel_size=(3, 3), strides=1, padding='SAME'
      )(conv_out)
      conv_out = nn.relu(conv_out)
      conv_out = nn.Conv(
          features=self.num_ch, kernel_size=(3, 3), strides=1, padding='SAME'
      )(conv_out)
      conv_out += block_input

    return conv_out


@gin.configurable
class ImpalaEncoder(nn.Module):
  """Impala Network which also outputs penultimate representation layers."""

  nn_scale: int = 1
  stack_sizes: Tuple[int, ...] = (16, 32, 32)
  num_blocks: int = 2

  def setup(self):
    logging.info('\t Creating %s ...', self.__class__.__name__)
    logging.info('\t Creating ImpalaDQNNetwork ...')
    logging.info('\t num_blocks: %s', self.num_blocks)
    logging.info('\t nn_scale: %s', self.nn_scale)
    logging.info('\t stack_sizes: %s', self.stack_sizes)
    self._stacks = [
        Stack(num_ch=stack_size * self.nn_scale, num_blocks=self.num_blocks)
        for stack_size in self.stack_sizes
    ]

  @nn.compact
  def __call__(self, x):
    for stack in self._stacks:
      x = stack(x)
    return nn.relu(x)


### DQN Network with ImpalaEncoder ###
@gin.configurable
class ImpalaDQNNetwork(nn.Module):
  """The convolutional network used to compute the agent's Q-values."""

  num_actions: int
  inputs_preprocessed: bool = False
  nn_scale: int = 1

  def setup(self):
    self.encoder = ImpalaEncoder(nn_scale=self.nn_scale)

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    if not self.inputs_preprocessed:
      x = preprocess_atari_inputs(x)
    x = self.encoder(x)
    x = x.reshape((-1))  # flatten
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    q_values = nn.Dense(features=self.num_actions, kernel_init=initializer)(x)
    return atari_lib.DQNNetworkType(q_values)


### DQN Networks ###
@gin.configurable
class NatureDQNNetwork(nn.Module):
  """The convolutional network used to compute the agent's Q-values."""

  num_actions: int
  inputs_preprocessed: bool = False

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    if not self.inputs_preprocessed:
      x = preprocess_atari_inputs(x)
    x = nn.Conv(
        features=32, kernel_size=(8, 8), strides=(4, 4), kernel_init=initializer
    )(x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64, kernel_size=(4, 4), strides=(2, 2), kernel_init=initializer
    )(x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64, kernel_size=(3, 3), strides=(1, 1), kernel_init=initializer
    )(x)
    x = nn.relu(x)
    x = x.reshape((-1))  # flatten
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    q_values = nn.Dense(features=self.num_actions, kernel_init=initializer)(x)
    return atari_lib.DQNNetworkType(q_values)


@gin.configurable
class ClassicControlDQNNetwork(nn.Module):
  """Jax DQN network for classic control environments."""

  num_actions: int
  num_layers: int = 2
  hidden_units: int = 512
  min_vals: Union[None, Tuple[float, ...]] = None
  max_vals: Union[None, Tuple[float, ...]] = None
  inputs_preprocessed: bool = False

  def setup(self):
    if self.min_vals is not None:
      assert self.max_vals is not None
      self._min_vals = jnp.array(self.min_vals)
      self._max_vals = jnp.array(self.max_vals)
    initializer = nn.initializers.xavier_uniform()
    self.layers = [
        nn.Dense(features=self.hidden_units, kernel_init=initializer)
        for _ in range(self.num_layers)
    ]
    self.final_layer = nn.Dense(
        features=self.num_actions, kernel_init=initializer
    )

  def __call__(self, x):
    if not self.inputs_preprocessed:
      x = x.astype(jnp.float32)
      x = x.reshape((-1))  # flatten
      if self.min_vals is not None:
        x -= self._min_vals
        x /= self._max_vals - self._min_vals
        x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    for layer in self.layers:
      x = layer(x)
      x = nn.relu(x)
    q_values = self.final_layer(x)
    return atari_lib.DQNNetworkType(q_values)


class FourierBasis(object):
  """Fourier Basis linear function approximation.

  Requires the ranges for each dimension, and is thus able to use only sine or
  cosine (and uses cosine). So, this has half the coefficients that a full
  Fourier approximation would use.

  Adapted from Will Dabney's (wdabney@) TF implementation for JAX.

  From the paper:
  G.D. Konidaris, S. Osentoski and P.S. Thomas. (2011)
  Value Function Approximation in Reinforcement Learning using the Fourier Basis
  """

  def __init__(
      self,
      nvars: int,
      min_vals: Union[float, Sequence[float]] = 0.0,
      max_vals: Optional[Union[float, Sequence[float]]] = None,
      order: int = 3,
  ):
    self.order = order
    self.min_vals = jnp.array(min_vals)
    self.max_vals = max_vals
    terms = itertools.product(range(order + 1), repeat=nvars)
    if max_vals is not None:
      assert len(self.min_vals) == len(self.max_vals)
      self.max_vals = jnp.array(self.max_vals)
      self.denominator = [
          max_vals[i] - min_vals[i] for i in range(len(min_vals))
      ]

    # Removing first iterate because it corresponds to the constant bias
    self.multipliers = jnp.array([list(map(int, x)) for x in terms][1:])

  def scale(self, values):
    shifted = values - self.min_vals
    if self.max_vals is None:
      return shifted

    return [shifted[i] / self.denominator[i] for i in range(len(shifted))]

  def compute_features(self, features):
    # Important to rescale features to be between [0,1]
    scaled = jnp.array(self.scale(features))
    return jnp.cos(jnp.pi * jnp.matmul(scaled, jnp.transpose(self.multipliers)))


@gin.configurable
class JaxFourierDQNNetwork(nn.Module):
  """Fourier-basis for DQN-like agents."""

  num_actions: int
  min_vals: Optional[Sequence[float]] = None
  max_vals: Optional[Sequence[float]] = None
  fourier_basis_order: int = 3

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    x = x.astype(jnp.float32)
    x = x.reshape((-1))  # flatten
    x = FourierBasis(
        x.shape[-1],
        self.min_vals,
        self.max_vals,
        order=self.fourier_basis_order,
    ).compute_features(x)
    q_values = nn.Dense(
        features=self.num_actions, kernel_init=initializer, use_bias=False
    )(x)
    return atari_lib.DQNNetworkType(q_values)


### Rainbow Networks ###
@gin.configurable
class RainbowNetwork(nn.Module):
  """Convolutional network used to compute the agent's return distributions."""

  num_actions: int
  num_atoms: int
  inputs_preprocessed: bool = False

  @nn.compact
  def __call__(self, x, support):
    initializer = nn.initializers.variance_scaling(
        scale=1.0 / jnp.sqrt(3.0), mode='fan_in', distribution='uniform'
    )
    if not self.inputs_preprocessed:
      x = preprocess_atari_inputs(x)
    x = nn.Conv(
        features=32, kernel_size=(8, 8), strides=(4, 4), kernel_init=initializer
    )(x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64, kernel_size=(4, 4), strides=(2, 2), kernel_init=initializer
    )(x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64, kernel_size=(3, 3), strides=(1, 1), kernel_init=initializer
    )(x)
    x = nn.relu(x)
    x = x.reshape((-1))  # flatten
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Dense(
        features=self.num_actions * self.num_atoms, kernel_init=initializer
    )(x)
    logits = x.reshape((self.num_actions, self.num_atoms))
    probabilities = nn.softmax(logits)
    q_values = jnp.sum(support * probabilities, axis=1)
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)


@gin.configurable
class ClassicControlRainbowNetwork(nn.Module):
  """Jax Rainbow network for classic control environments."""

  num_actions: int
  num_atoms: int
  num_layers: int = 2
  hidden_units: int = 512
  min_vals: Union[None, Tuple[float, ...]] = None
  max_vals: Union[None, Tuple[float, ...]] = None
  inputs_preprocessed: bool = False

  def setup(self):
    if self.min_vals is not None:
      self._min_vals = jnp.array(self.min_vals)
      self._max_vals = jnp.array(self.max_vals)
    initializer = nn.initializers.xavier_uniform()
    self.layers = [
        nn.Dense(features=self.hidden_units, kernel_init=initializer)
        for _ in range(self.num_layers)
    ]
    self.final_layer = nn.Dense(
        features=self.num_actions * self.num_atoms, kernel_init=initializer
    )

  def __call__(self, x, support):
    if not self.inputs_preprocessed:
      x = x.astype(jnp.float32)
      x = x.reshape((-1))  # flatten
      if self.min_vals is not None:
        x -= self._min_vals
        x /= self._max_vals - self._min_vals
        x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    for layer in self.layers:
      x = layer(x)
      x = nn.relu(x)
    x = self.final_layer(x)
    logits = x.reshape((self.num_actions, self.num_atoms))
    probabilities = nn.softmax(logits)
    q_values = jnp.sum(support * probabilities, axis=1)
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)


### Implicit Quantile Networks ###
class ImplicitQuantileNetwork(nn.Module):
  """The Implicit Quantile Network (Dabney et al., 2018).."""

  num_actions: int
  quantile_embedding_dim: int
  inputs_preprocessed: bool = False

  @nn.compact
  def __call__(self, x, num_quantiles, rng):
    initializer = nn.initializers.variance_scaling(
        scale=1.0 / jnp.sqrt(3.0), mode='fan_in', distribution='uniform'
    )
    if not self.inputs_preprocessed:
      x = preprocess_atari_inputs(x)
    x = nn.Conv(
        features=32, kernel_size=(8, 8), strides=(4, 4), kernel_init=initializer
    )(x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64, kernel_size=(4, 4), strides=(2, 2), kernel_init=initializer
    )(x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64, kernel_size=(3, 3), strides=(1, 1), kernel_init=initializer
    )(x)
    x = nn.relu(x)
    x = x.reshape((-1))  # flatten
    state_vector_length = x.shape[-1]
    state_net_tiled = jnp.tile(x, [num_quantiles, 1])
    quantiles_shape = [num_quantiles, 1]
    quantiles = jax.random.uniform(rng, shape=quantiles_shape)
    quantile_net = jnp.tile(quantiles, [1, self.quantile_embedding_dim])
    quantile_net = (
        jnp.arange(1, self.quantile_embedding_dim + 1, 1).astype(jnp.float32)
        * onp.pi
        * quantile_net
    )
    quantile_net = jnp.cos(quantile_net)
    quantile_net = nn.Dense(
        features=state_vector_length, kernel_init=initializer
    )(quantile_net)
    quantile_net = nn.relu(quantile_net)
    x = state_net_tiled * quantile_net
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    quantile_values = nn.Dense(
        features=self.num_actions, kernel_init=initializer
    )(x)
    return atari_lib.ImplicitQuantileNetworkType(quantile_values, quantiles)


### Quantile Networks ###
@gin.configurable
class QuantileNetwork(nn.Module):
  """Convolutional network used to compute the agent's return quantiles."""

  num_actions: int
  num_atoms: int
  inputs_preprocessed: bool = False

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.variance_scaling(
        scale=1.0 / jnp.sqrt(3.0), mode='fan_in', distribution='uniform'
    )
    if not self.inputs_preprocessed:
      x = preprocess_atari_inputs(x)
    x = nn.Conv(
        features=32, kernel_size=(8, 8), strides=(4, 4), kernel_init=initializer
    )(x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64, kernel_size=(4, 4), strides=(2, 2), kernel_init=initializer
    )(x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64, kernel_size=(3, 3), strides=(1, 1), kernel_init=initializer
    )(x)
    x = nn.relu(x)
    x = x.reshape((-1))  # flatten
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Dense(
        features=self.num_actions * self.num_atoms, kernel_init=initializer
    )(x)
    logits = x.reshape((self.num_actions, self.num_atoms))
    probabilities = nn.softmax(logits)
    q_values = jnp.mean(logits, axis=1)
    return atari_lib.RainbowNetworkType(q_values, logits, probabilities)


### Noisy Nets for FullRainbowNetwork ###
@gin.configurable
class NoisyNetwork(nn.Module):
  """Noisy Network from Fortunato et al. (2018).

  Attributes:
    rng_key: jax.interpreters.xla.DeviceArray, key for JAX RNG.
    eval_mode: bool, whether to turn off noise during evaluation.
  """

  rng_key: jax.Array
  eval_mode: bool = False

  @staticmethod
  def sample_noise(key, shape):
    return jax.random.normal(key, shape)

  @staticmethod
  def f(x):
    # See (10) and (11) in Fortunato et al. (2018).
    return jnp.multiply(jnp.sign(x), jnp.power(jnp.abs(x), 0.5))

  @nn.compact
  def __call__(self, x, features, bias=True, kernel_init=None):
    def mu_init(key, shape):
      # Initialization of mean noise parameters (Section 3.2)
      low = -1 / jnp.power(x.shape[0], 0.5)
      high = 1 / jnp.power(x.shape[0], 0.5)
      return jax.random.uniform(key, minval=low, maxval=high, shape=shape)

    def sigma_init(key, shape, dtype=jnp.float32):  # pylint: disable=unused-argument
      # Initialization of sigma noise parameters (Section 3.2)
      return jnp.ones(shape, dtype) * (0.1 / onp.sqrt(x.shape[0]))

    if self.eval_mode:
      # Turn off noise during evaluation
      w_epsilon = onp.zeros(shape=(x.shape[0], features), dtype=onp.float32)
      b_epsilon = onp.zeros(shape=(features,), dtype=onp.float32)
    else:
      # Factored gaussian noise in (10) and (11) in Fortunato et al. (2018).
      rng_p, rng_q = jax.random.split(self.rng_key, num=2)
      p = NoisyNetwork.sample_noise(rng_p, [x.shape[0], 1])
      q = NoisyNetwork.sample_noise(rng_q, [1, features])
      f_p = NoisyNetwork.f(p)
      f_q = NoisyNetwork.f(q)
      w_epsilon = f_p * f_q
      b_epsilon = jnp.squeeze(f_q)

    # See (8) and (9) in Fortunato et al. (2018) for output computation.
    w_mu = self.param('kernel_mu', mu_init, (x.shape[0], features))
    w_sigma = self.param('kernel_sigma', sigma_init, (x.shape[0], features))
    w = w_mu + jnp.multiply(w_sigma, w_epsilon)
    ret = jnp.matmul(x, w)

    b_mu = self.param('bias_mu', mu_init, (features,))
    b_sigma = self.param('bias_sigma', sigma_init, (features,))
    b = b_mu + jnp.multiply(b_sigma, b_epsilon)
    return jnp.where(bias, ret + b, ret)


### FullRainbowNetwork ###
def feature_layer(key, noisy, eval_mode=False):
  """Network feature layer depending on whether noisy_nets are used on or not."""

  def noisy_net(x, features):
    return NoisyNetwork(rng_key=key, eval_mode=eval_mode)(x, features)

  def dense_net(x, features):
    return nn.Dense(features, kernel_init=nn.initializers.xavier_uniform())(x)

  return noisy_net if noisy else dense_net


@gin.configurable
class FullRainbowNetwork(nn.Module):
  """Jax Rainbow network for Full Rainbow.

  Attributes:
    num_actions: int, number of actions the agent can take at any state.
    num_atoms: int, the number of buckets of the value function distribution.
    noisy: bool, Whether to use noisy networks.
    dueling: bool, Whether to use dueling network architecture.
    distributional: bool, whether to use distributional RL.
  """

  num_actions: int
  num_atoms: int
  noisy: bool = True
  dueling: bool = True
  distributional: bool = True
  inputs_preprocessed: bool = False

  @nn.compact
  def __call__(self, x, support, eval_mode=False, key=None):
    # Generate a random number generation key if not provided
    if key is None:
      key = jax.random.PRNGKey(int(time.time() * 1e6))

    if not self.inputs_preprocessed:
      x = preprocess_atari_inputs(x)

    hidden_sizes = [32, 64, 64]
    kernel_sizes = [8, 4, 3]
    stride_sizes = [4, 2, 1]
    for hidden_size, kernel_size, stride_size in zip(
        hidden_sizes, kernel_sizes, stride_sizes
    ):
      x = nn.Conv(
          features=hidden_size,
          kernel_size=(kernel_size, kernel_size),
          strides=(stride_size, stride_size),
          kernel_init=nn.initializers.xavier_uniform(),
      )(x)
      x = nn.relu(x)
    x = x.reshape((-1))  # flatten

    net = feature_layer(key, self.noisy, eval_mode=eval_mode)
    x = net(x, features=512)  # Single hidden layer of size 512
    x = nn.relu(x)

    if self.dueling:
      adv = net(x, features=self.num_actions * self.num_atoms)
      value = net(x, features=self.num_atoms)
      adv = adv.reshape((self.num_actions, self.num_atoms))
      value = value.reshape((1, self.num_atoms))
      logits = value + (adv - (jnp.mean(adv, axis=0, keepdims=True)))
    else:
      x = net(x, features=self.num_actions * self.num_atoms)
      logits = x.reshape((self.num_actions, self.num_atoms))

    if self.distributional:
      probabilities = nn.softmax(logits)
      q_values = jnp.sum(support * probabilities, axis=1)
      return atari_lib.RainbowNetworkType(q_values, logits, probabilities)
    q_values = jnp.sum(logits, axis=1)  # Sum over all the num_atoms
    return atari_lib.DQNNetworkType(q_values)


@gin.configurable
class PPOSharedNetwork(nn.Module):
  """Shared network for PPO actor and critic."""

  inputs_preprocessed: bool = False

  @nn.compact
  def __call__(self, x) -> jnp.ndarray:
    initializer = nn.initializers.orthogonal(jnp.sqrt(2))
    if not self.inputs_preprocessed:
      x = preprocess_atari_inputs(x)
    x = nn.Conv(
        features=32,
        kernel_size=(8, 8),
        strides=4,
        padding='VALID',
        kernel_init=initializer,
    )(x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64,
        kernel_size=(4, 4),
        strides=2,
        padding='VALID',
        kernel_init=initializer,
    )(x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64,
        kernel_size=(3, 3),
        strides=1,
        padding='VALID',
        kernel_init=initializer,
    )(x)
    x = nn.relu(x)
    x = x.reshape((-1))  # flatten
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    return x


@gin.configurable
class PPOActorNetwork(nn.Module):
  """Actor backbone for PPO."""

  num_actions: int

  @nn.compact
  def __call__(self, x) -> jnp.ndarray:
    initializer = nn.initializers.orthogonal(0.01)
    return nn.Dense(features=self.num_actions, kernel_init=initializer)(x)


class PPOCriticNetwork(nn.Module):
  """Critic backbone for PPO."""

  @nn.compact
  def __call__(self, x) -> jnp.ndarray:
    initializer = nn.initializers.orthogonal(1.0)
    return nn.Dense(features=1, kernel_init=initializer)(x)


@gin.configurable
class PPODiscreteActorCriticNetwork(nn.Module):
  """Convolutional actor critic value and policy networks."""

  action_shape: Tuple[int, ...]
  inputs_preprocessed: bool = False

  def setup(self):
    action_dim = functools.reduce(operator.mul, self.action_shape, 1)
    self._shared_network = PPOSharedNetwork(self.inputs_preprocessed)
    self._actor = PPOActorNetwork(action_dim)
    self._critic = PPOCriticNetwork()

  def __call__(
      self, state: jnp.ndarray, key: jnp.ndarray
  ) -> continuous_networks.ActorCriticOutput:
    actor_output = self.actor(state, key)
    critic_output = self.critic(state)
    return continuous_networks.PPOActorCriticOutput(actor_output, critic_output)

  def actor(
      self,
      state: jnp.ndarray,
      key: Optional[jnp.ndarray] = None,
      action: Optional[jnp.ndarray] = None,
  ) -> continuous_networks.PPOActorOutput:
    logits = self._actor(self._shared_network(state))
    dist = tfp.distributions.Categorical(logits=logits)
    if action is None:
      if key is None:
        raise ValueError('Key must be provided if action is None.')
      action = dist.sample(seed=key)
    log_probability = dist.log_prob(action)
    entropy = dist.entropy()
    return continuous_networks.PPOActorOutput(action, log_probability, entropy)

  def critic(self, state: jnp.ndarray) -> continuous_networks.PPOCriticOutput:
    return continuous_networks.PPOCriticOutput(
        self._critic(self._shared_network(state))
    )

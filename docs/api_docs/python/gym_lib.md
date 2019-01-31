<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gym_lib" />
<meta itemprop="path" content="Stable" />
</div>

# Module: gym_lib

Gym-specific (non-Atari) utilities.

Some network specifications specific to certain Gym environments are provided
here.

Includes a wrapper class around Gym environments. This class makes general Gym
environments conformant with the API Dopamine is expecting.

## Classes

[`class GymPreprocessing`](./gym_lib/GymPreprocessing.md): A Wrapper class
around Gym environments.

## Functions

[`acrobot_dqn_network(...)`](./gym_lib/acrobot_dqn_network.md): Builds the deep
network used to compute the agent's Q-values.

[`acrobot_rainbow_network(...)`](./gym_lib/acrobot_rainbow_network.md): Build
the deep network used to compute the agent's Q-value distributions.

[`cartpole_dqn_network(...)`](./gym_lib/cartpole_dqn_network.md): Builds the
deep network used to compute the agent's Q-values.

[`cartpole_rainbow_network(...)`](./gym_lib/cartpole_rainbow_network.md): Build
the deep network used to compute the agent's Q-value distributions.

[`create_gym_environment(...)`](./gym_lib/create_gym_environment.md): Wraps a
Gym environment with some basic preprocessing.

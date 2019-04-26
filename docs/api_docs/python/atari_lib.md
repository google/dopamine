<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="atari_lib" />
<meta itemprop="path" content="Stable" />
</div>

# Module: atari_lib

Atari-specific utilities including Atari-specific network architectures.

This includes a class implementing minimal Atari 2600 preprocessing, which is in
charge of: . Emitting a terminal signal when losing a life (optional). . Frame
skipping and color pooling. . Resizing the image before it is provided to the
agent.

## Classes

[`class AtariPreprocessing`](./atari_lib/AtariPreprocessing.md): A class
implementing image preprocessing for Atari 2600 agents.

## Functions

[`create_atari_environment(...)`](./atari_lib/create_atari_environment.md):
Wraps an Atari 2600 Gym environment with some basic preprocessing.

[`implicit_quantile_network(...)`](./atari_lib/implicit_quantile_network.md):
The Implicit Quantile ConvNet.

[`nature_dqn_network(...)`](./atari_lib/nature_dqn_network.md): The
convolutional network used to compute the agent's Q-values.

[`rainbow_network(...)`](./atari_lib/rainbow_network.md): The convolutional
network used to compute agent's Q-value distributions.

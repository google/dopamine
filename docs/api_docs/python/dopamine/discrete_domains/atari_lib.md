description: Atari-specific utilities including Atari-specific network
architectures.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.discrete_domains.atari_lib" />
<meta itemprop="path" content="Stable" />
</div>

# Module: dopamine.discrete_domains.atari_lib

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/discrete_domains/atari_lib.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Atari-specific utilities including Atari-specific network architectures.

This includes a class implementing minimal Atari 2600 preprocessing, which is in
charge of: . Emitting a terminal signal when losing a life (optional). . Frame
skipping and color pooling. . Resizing the image before it is provided to the
agent.

## Networks

We are subclassing keras.models.Model in our network definitions. Each network
class has two main functions: `.__init__` and `.call`. When we create our
network the `__init__` function is called and necessary layers are defined. Once
we create our network, we can create the output operations by doing `call`s to
our network with different inputs. At each call, the same parameters will be
used.

More information about keras.Model API can be found here:
https://www.tensorflow.org/api_docs/python/tf/keras/models/Model

## Network Types

Network types are namedtuples that define the output signature of the networks
used. Please use the appropriate signature as needed when defining new networks.

## Classes

[`class AtariPreprocessing`](../../dopamine/discrete_domains/atari_lib/AtariPreprocessing.md):
A class implementing image preprocessing for Atari 2600 agents.

## Functions

[`create_atari_environment(...)`](../../dopamine/discrete_domains/atari_lib/create_atari_environment.md):
Wraps an Atari 2600 Gym environment with some basic preprocessing.

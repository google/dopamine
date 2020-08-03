# JAX

This directory contains all the code that is specific to the
[JAX](https://github.com/google/jax) version of Dopamine.

## Why JAX?
We feel the JAX philosophy goes very well with that of Dopamine: flexibility for
research without sacrificing simplicity. We have been using it for our research
for a little while now and have found its modularity quite appealing in terms of
simplifying some of the more difficult aspects of the agents.

Consider, for instance, the projection operator
used for the C51 and Rainbow agents, which is a rather complex TensorFlow op due
to the necessary dynamic indexing. Excluding comments and shape assertions,
this went from
[23 lines of fairly complex TensorFlow code](https://github.com/google/dopamine/blob/master/dopamine/agents/rainbow/rainbow_agent.py#L330)
to
[9 lines of JAX
code](https://github.com/google/dopamine/blob/master/dopamine/jax/agents/rainbow/rainbow_agent.py#L331)
that is a more straightforward implementation and easier to relate to the way it
was presented in the paper.

We are able to achieve this simplicity in large part because of JAX's
[vmap](https://jax.readthedocs.io/en/latest/jax.html?highlight=vmap#vectorization-vmap)
functionality, which helps us avoid explicitly dealing with batch dimensions.

Thanks to JAX's [Just in Time
compilation](https://jax.readthedocs.io/en/latest/jax.html?highlight=jit#just-in-time-compilation-jit)
our JAX agents achieve the same speed of training as our TensorFlow agents.

## Code organization

The file `networks.py` contains the network definitions for all of the JAX
agents, while the `agents/` subdirectory mimics the top-level `agents/`
directory for the original TensorFlow agents.

## Trained agent checkpoints
We provide checkpoitns for trained JAX agents with the following URL format:

```
https://storage.cloud.google.com/download-dopamine-rl/jax/{AGENT}/{GAME}/{RUN}/ckpt.199
```

where the possible values are:

*  `AGENT`: `['dqn', 'c51', 'rainbow', 'quantile', 'implicit_quantile']`
*  `GAME`: `['Pong', 'SpaceInvaders', 'Seaquest', 'Qbert', 'Breakout', 'Asterix']`
*  `RUN`: `[1, 2, 3, 4, 5]`

## Colaboratory notebook
We've also added a
[colab notebook](https://colab.research.google.com/github/google/dopamine/blob/master/dopamine/colab/jax_agent_visualizer.ipynb)
to visualize trained JAX agents.

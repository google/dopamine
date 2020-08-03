# Jax

This directory contains all the code that is specific to the
[Jax](https://github.com/google/jax) version of Dopamine.

## Code organization

The file `networks.py` contains the network definitions for all of the Jax
agents, while the `agents/` subdirectory mimics the top-level `agents/`
directory for the original TensorFlow agents.

## Trained agent checkpoints
We provide checkpoitns for trained Jax agents with the following URL format:

```
https://storage.cloud.google.com/download-dopamine-rl/jax/{AGENT}/{GAME}/{RUN}/ckpt.199
```

where the possible values are:

*  `AGENT`: `['dqn', 'c51', 'rainbow', 'quantile', 'implicit_quantile']`
*  `GAME`: `['Pong', 'SpaceInvaders', 'Seaquest', 'Qbert', 'Breakout', 'Asterix']`
*  `RUN`: `[1, 2, 3, 4, 5]`

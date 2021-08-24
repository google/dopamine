# Baseline data

This directory provides information about the baseline data provided by
Dopamine. We currently only support SAC for mujoco. Also, the baseline data
is reported using the training regime, not evaluation. For SAC, that means
we are using sampled actions, not the mean action.

The default configuration file (set up with
[gin configuration framework](https://github.com/google/gin-config)) is:

*   [`dopamine/jax/agents/sac/configs/sac.gin`](https://github.com/google/dopamine/blob/master/dopamine/jax/agents/sac/configs/sac.gin)

## Visualization
We provide a [website](https://google.github.io/dopamine/baselines/mujoco/plots.html)
where you can quickly visualize the training run for SAC.

The plots are rendered from a set of
[JSON files](https://github.com/google/dopamine/tree/master/baselines/mujoco/data)
which we compiled. These may prove useful in their own right to compare
against results obtained from other frameworks.


[sac]: https://arxiv.org/abs/1812.05905

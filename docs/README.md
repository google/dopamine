# Overview

This document gives examples and pointers on how to experiment with and extend
Dopamine.

You can find the documentation for each module in our codebase in our
[API documentation](https://github.com/google/dopamine/blob/master/docs/api_docs/python/index.md).

## File organization

Dopamine is organized as follows:

*   [`jax`](https://github.com/google/dopamine/tree/master/dopamine/jax)
    contains jax agent implementations and networks.
*   [`agents`](https://github.com/google/dopamine/tree/master/dopamine/agents)
    contains tenforflow agent implementations.
*   [`atari`](https://github.com/google/dopamine/tree/master/dopamine/atari)
    contains Atari-specific code, including code to run experiments and
    preprocessing code.
*   [`common`](https://github.com/google/dopamine/tree/master/dopamine/common)
    contains additional helper functionality, including logging and
    checkpointing.
*   [`replay_memory`](https://github.com/google/dopamine/tree/master/dopamine/replay_memory)
    contains the replay memory schemes used in Dopamine.
*   [`colab`](https://github.com/google/dopamine/tree/master/dopamine/colab)
    contains code used to inspect the results of experiments, as well as example
    colab notebooks.
*   [`tests`](https://github.com/google/dopamine/tree/master/tests) contains all
    our test files.

## Training agents

### Atari games

The entry point to the standard Atari 2600 experiment is
[`dopamine/discrete_domains/train.py`](https://github.com/google/dopamine/blob/master/dopamine/discrete_domains/train.py).
To run the basic DQN agent,

```
python -um dopamine.discrete_domains.train \
  --base_dir /tmp/dopamine_runs \
  --gin_files dopamine/agents/dqn/configs/dqn.gin
```

By default, this will kick off an experiment lasting 200 million frames. The
command-line interface will output statistics about the latest training episode:

```
[...]
I0824 17:13:33.078342 140196395337472 tf_logging.py:115] gamma: 0.990000
I0824 17:13:33.795608 140196395337472 tf_logging.py:115] Beginning training...
Steps executed: 5903 Episode length: 1203 Return: -19.
```

To get finer-grained information about the process, you can adjust the
experiment parameters in
[`dopamine/agents/dqn/configs/dqn.gin`](https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/configs/dqn.gin),
in particular by reducing `Runner.training_steps` and `Runner.evaluation_steps`,
which together determine the total number of steps needed to complete an
iteration. This is useful if you want to inspect log files or checkpoints, which
are generated at the end of each iteration.

More generally, the whole of Dopamine is easily configured using the
[gin configuration framework](https://github.com/google/gin-config).

### Non-Atari discrete environments

We provide sample configuration files for training an agent on Cartpole and
Acrobot. For example, to train C51 on Cartpole with default settings, run the
following command:

```
python -um dopamine.discrete_domains.train \
  --base_dir /tmp/dopamine_runs \
  --gin_files dopamine/agents/rainbow/configs/c51_cartpole.gin
```

You can train Rainbow on Acrobot with the following command:

```
python -um dopamine.discrete_domains.train \
  --base_dir /tmp/dopamine_runs \
  --gin_files dopamine/agents/rainbow/configs/rainbow_acrobot.gin
```

### Continuous control environments

The entry point for continuous control agents is
[`dopamine/continuous_domains/train.py`](https://github.com/google/dopamine/blob/master/dopamine/continuous_domains/train.py).
You will need a Mujoco key to run the following example. To run SAC on the
HalfCheetah environment of Mujoco, run:

```
python -um dopamine.continuous_domains.train \
  --base_dir /tmp/dopamine_runs \
  --gin_files dopamine/jax/agents/sac/configs/sac.gin
```

By default, this will kick off an experiment lasting 3200 episodes, with 1000
environment steps per episode. The command-line interface will output statistics
about the latest training episode:

```
[...]
I0908 17:19:39.618797 1803949 run_experiment.py:446] Starting iteration 0
I0908 17:19:40.592262 1803949 run_experiment.py:405] Average undiscounted return per training episode: -168.19
I0908 17:19:40.592391 1803949 run_experiment.py:407] Average training steps per second: 1027.80
I0908 17:19:45.699378 1803949 run_experiment.py:427] Average undiscounted return per evaluation episode: -279.07
```

To run with different environments/hyperparemeters, adjust the gin config file
found here:
[`dopamine/jax/agents/sac/configs/sac.gin`](https://github.com/google/dopamine/blob/master/dopamine/jax/agents/sac/configs/sac.gin).
For your experiments, you may choose to supply a new gin config file, or
override the existing config file with command line `gin_bindings` args.

For more information on using gin, see the
[gin github repo](https://github.com/google/gin-config).

## Configuring agents

The whole of Dopamine is easily configured using the
[gin configuration framework](https://github.com/google/gin-config).

We provide a number of configuration files for each of the agents. The main
configuration file for each agent corresponds to an "apples to apples"
comparison, where hyperparameters have been selected to give a standardized
performance comparison between agents. These are

*   [`dopamine/agents/dqn/configs/dqn.gin`](https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/configs/dqn.gin)
*   [`dopamine/agents/rainbow/configs/c51.gin`](https://github.com/google/dopamine/blob/master/dopamine/agents/rainbow/configs/c51.gin)
*   [`dopamine/agents/rainbow/configs/rainbow.gin`](https://github.com/google/dopamine/blob/master/dopamine/agents/rainbow/configs/rainbow.gin)
*   [`dopamine/agents/implicit_quantile/configs/implicit_quantile.gin`](https://github.com/google/dopamine/blob/master/dopamine/agents/implicit_quantile/configs/implicit_quantile.gin)

More details on the exact choices behind these parameters are given in our
[baselines page](https://github.com/google/dopamine/tree/master/baselines/).

We also provide configuration files corresponding to settings previously used in
the literature. These are

*   [`dopamine/agents/dqn/configs/dqn_nature.gin`](https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/configs/dqn_nature.gin)
    ([Mnih et al., 2015][dqn])
*   [`dopamine/agents/dqn/configs/dqn_icml.gin`](https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/configs/dqn_icml.gin)
    ([Bellemare et al., 2017][c51])
*   [`dopamine/agents/rainbow/configs/c51_icml.gin`](https://github.com/google/dopamine/blob/master/dopamine/agents/rainbow/configs/c51_icml.gin)
    ([Bellemare et al., 2017][c51])
*   [`dopamine/agents/implicit_quantile/configs/implicit_quantile_icml.gin`](https://github.com/google/dopamine/blob/master/dopamine/agents/implicit_quantile/configs/implicit_quantile_icml.gin)
    ([Dabney et al., 2018][iqn])

All of these use the deterministic version of the Arcade Learning Environment
(ALE), and slightly different hyperparameters.

## Checkpointing and logging

Dopamine provides basic functionality for performing experiments. This
functionality can be broken down into two main components: *checkpointing* and
*logging*. Both components depend on the command-line parameter `base_dir`,
which informs Dopamine of where it should store experimental data.

#### Checkpointing

By default, Dopamine will save an experiment checkpoint every iteration: one
training and one evaluation phase, following a standard set by [Mnih et al][dqn].
Checkpoints are saved in the `checkpoints` subdirectory under `base_dir`. At a
high-level, the following are checkpointed:

*   Experiment statistics (number of iterations performed, learning curves,
    etc.). This happens in
    [`dopamine/atari/run_experiment.py`](https://github.com/google/dopamine/blob/master/dopamine/atari/run_experiment.py),
    in the method
    [`run_experiment`](https://github.com/google/dopamine/blob/master/docs/api_docs/python/run_experiment/TrainRunner.md#run_experiment).
*   Agent variables, including the tensorflow graph. This happens in
    [`dopamine/agents/dqn/dqn_agent.py`](https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/dqn_agent.py),
    in the methods
    [`bundle_and_checkpoint`](https://github.com/google/dopamine/blob/master/docs/api_docs/python/dqn_agent/DQNAgent.md#bundle_and_checkpoint)
    and
    [`unbundle`](https://github.com/google/dopamine/blob/master/docs/api_docs/python/dqn_agent/DQNAgent.md#unbundle).
*   Replay buffer data. Atari 2600 replay buffers have a large memory footprint.
    As a result, Dopamine uses additional code to keep memory usage low. The
    relevant methods are found in
    [`dopamine/agents/replay_memory/circular_replay_buffer.py`](https://github.com/google/dopamine/blob/master/dopamine/replay_memory/circular_replay_buffer.py),
    and are called
    [`save`](https://github.com/google/dopamine/blob/master/docs/api_docs/python/circular_replay_buffer/OutOfGraphReplayBuffer.md#save)
    and
    [`load`](https://github.com/google/dopamine/blob/master/docs/api_docs/python/circular_replay_buffer/OutOfGraphReplayBuffer.md#load).

If you're curious, the checkpointing code itself is in
[`dopamine/common/checkpointer.py`](https://github.com/google/dopamine/blob/master/dopamine/common/checkpointer.py).

#### Logging

At the end of each iteration, Dopamine also records the agent's performance,
both during training and (if enabled) during an optional evaluation phase. The
log files are generated in
[`dopamine/atari/run_experiment.py`](https://github.com/google/dopamine/blob/master/dopamine/atari/run_experiment.py)
and more specifically in
[`dopamine/common/logger.py`](https://github.com/google/dopamine/blob/master/dopamine/common/logger.py),
and are pickle files containing a dictionary mapping iteration keys
(e.g., `"iteration_47"`) to dictionaries containing data.

A simple way to read log data from multiple experiments is to use the provided
[`read_experiment`](https://github.com/google/dopamine/blob/master/docs/api_docs/python/utils/read_experiment.md)
method in
[`colab/utils.py`](https://github.com/google/dopamine/blob/master/dopamine/colab/utils.py).

We provide a
[colab](https://colab.research.google.com/github/google/dopamine/blob/master/dopamine/colab/load_statistics.ipynb)
to illustrate how you can load the statistics from an experiment and plot them
against our provided baseline runs.

## Modifying and extending agents

Dopamine is designed to make algorithmic research simple. With this in mind, we
decided to keep a relatively flat class hierarchy, with no abstract base class;
we've found this sufficient for our research purposes, with the added benefits
of simplicity and ease of use. To begin, we recommend modifying the agent code
directly to suit your research purposes.

We provide a
[colab](https://colab.research.google.com/github/google/dopamine/blob/master/dopamine/colab/agents.ipynb)
where we illustrate how one can extend the DQN agent, or create a new agent from
scratch, and then plot the experimental results against our provided baselines.

#### DQN

The DQN agent is contained in two files:

*   The *agent class*, in
    [`dopamine/agents/dqn/dqn_agent.py`](https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/dqn_agent.py).
*   The *replay buffer*, in
    [`dopamine/replay_memory/circular_replay_buffer.py`](https://github.com/google/dopamine/blob/master/dopamine/replay_memory/circular_replay_buffer.py).

The agent class defines the DQN network, the update rule, and also the basic
operations of a RL agent (epsilon-greedy action selection, storing transitions,
episode bookkeeping, etc.). For example, the Q-Learning update rule used in DQN
is defined in two methods, `_build_target_q_op` and `_build_train_op`.

#### Rainbow and C51

The Rainbow agent is contained in two files:

*   The agent class in
    [`dopamine/agents/rainbow/rainbow_agent.py`](https://github.com/google/dopamine/blob/master/dopamine/agents/rainbow/rainbow_agent.py),
    inheriting from the DQN agent.
*   The replay buffer in
    [`dopamine/replay_memory/prioritized_replay_buffer.py`](https://github.com/google/dopamine/blob/master/dopamine/replay_memory/prioritized_replay_buffer.py),
    inheriting from DQN's replay buffer.

The C51 agent is a specific parametrization of the Rainbow agent, where
`update_horizon` (the `n` in n-step update) is set to 1 and a uniform replay
scheme is used.

#### Implicit quantile networks (IQN)

The IQN agent is defined by one additional file:

*   [`dopamine/agents/implicit_quantile/implicit_quantile_agent.py`](https://github.com/google/dopamine/blob/master/dopamine/agents/implicit_quantile/implicit_quantile_agent.py),
    inheriting from the Rainbow agent.

## Downloads

We provide a series of files for all 4 agents on all 60 games. These are all
`*.tar.gz` files which you will need to uncompress:

*   The raw logs are available
    [here](https://storage.cloud.google.com/download-dopamine-rl/compiled_raw_logs_files.tar.gz)
    *  You can view this
       [colab](https://colab.research.google.com/github/google/dopamine/blob/master/dopamine/colab/load_statistics.ipynb)
       for instructions on how to load and visualize them.
*   The compiled pickle files are available
    [here](https://storage.cloud.google.com/download-dopamine-rl/compiled_pkl_files.tar.gz)
    *  We make use of these compiled pickle files in both
       [agents](https://colab.research.google.com/github/google/dopamine/blob/master/dopamine/colab/agents.ipynb)
       and the
       [statistics](https://colab.research.google.com/github/google/dopamine/blob/master/dopamine/colab/load_statistics.ipynb)
       colabs.
*   The Tensorboard event files are available
    [here](https://storage.cloud.google.com/download-dopamine-rl/compiled_tb_event_files.tar.gz)
    *  We provide a
       [colab](https://colab.research.google.com/github/google/dopamine/blob/master/dopamine/colab/tensorboard.ipynb)
       where you can start Tensorboard directly from the colab using `ngrok`.
       In the provided example your Tensorboard will look something like this:

<div align="center">
  <img src="https://google.github.io/dopamine/images/all_asterix_tb.png"><br><br>
</div>

    *  You can also view these with Tensorboard on your machine. For instance, after
       uncompressing the files you can run:

       ```
       tensorboard --logdir c51/Asterix/
       ```

       to display the training runs for C51 on Asterix:

<div align="center">
  <img src="https://google.github.io/dopamine/images/c51_asterix_tb.png"><br><br>
</div>

*   The TensorFlow checkpoint files for 5 independent runs of the 4 agents on
    all 60 games are available. The format for each of the files is:
    `https://storage.cloud.google.com/download-dopamine-rl/lucid/${AGENT}/${GAME}/${RUN}/tf_ckpt-199.${SUFFIX}`,
    where:
    *  `AGENT` can be "dqn", "c51", "rainbow", or "iqn".
    *  `GAME` can be any of the 60 games.
    *  `RUN` can be 1, 2, 3, 4, or 5
    *  `SUFFIX` can be one of `data-00000-of-00001`, `index`, or `meta`.
*   You can also download all of these as a single `.tar.gz` file. **Note**: these files are quite large, over 15Gb each.
    *  [DQN checkpoints](https://storage.cloud.google.com/download-dopamine-rl/dqn_checkpoints.tar.gz)
    *  [C51 checkpoints](https://storage.cloud.google.com/download-dopamine-rl/c51_checkpoints.tar.gz)
    *  [Rainbow checkpoints](https://storage.cloud.google.com/download-dopamine-rl/rainbow_checkpoints.tar.gz)
    *  [IQN checkpoints](https://storage.cloud.google.com/download-dopamine-rl/iqn_checkpoints.tar.gz)

[dqn]: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
[c51]: http://proceedings.mlr.press/v70/bellemare17a.html
[rainbow]: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17204/16680
[iqn]: https://arxiv.org/abs/1806.06923

# Baseline data

This directory provides information about the baseline data provided by
Dopamine. The default hyperparameter configuration for the 4 agents we are
providing yields a standardized "apples to apples" comparison between them.

The default configuration files files for each agent (set up with
[gin configuration framework](https://github.com/google/gin-config)) are:

*   [`dopamine/agents/dqn/configs/dqn.gin`](https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/configs/dqn.gin)
*   [`dopamine/agents/rainbow/configs/c51.gin`](https://github.com/google/dopamine/blob/master/dopamine/agents/rainbow/configs/c51.gin)
*   [`dopamine/agents/rainbow/configs/rainbow.gin`](https://github.com/google/dopamine/blob/master/dopamine/agents/rainbow/configs/rainbow.gin)
*   [`dopamine/agents/implicit_quantile/configs/implicit_quantile.gin`](https://github.com/google/dopamine/blob/master/dopamine/agents/implicit_quantile/configs/implicit_quantile.gin)

## Hyperparemeter comparison
Our results compare the agents with the same hyperparameters: target
network update frequency, frequency at which exploratory actions are selected (ε), the
length of the schedule over which ε is annealed, and the number of agent steps
before training occurs. Changing these parameters can significantly affect
performance, without necessarily being indicative of an algorithmic difference.
Unsurprisingly, DQN performs much better when trained with 1% of exploratory
actions instead of 10% (as used in the original Nature paper). Step size and
optimizer were taken as published. The table below summarizes our choices. All
numbers are in ALE frames.

|                                     | Our baseline results | [DQN][dqn]       | [C51][c51]       | [Rainbow][rainbow] | [IQN][iqn]       |
| :---------------------------------- | :------------------: | :--------:       | :--------:       | :----------------: | :--------:       |
| **Training ε**                      | 0.01                 | 0.1              | 0.01             | 0.01               | 0.01             |
| **Evaluation ε**                    | 0.001                | 0.01             | 0.001            | *                  | 0.001            |
| **ε decay schedule**                | 1,000,000 frames     | 4,000,000 frames | 4,000,000 frames | 1,000,000 frames   | 4,000,000 frames |
| **Min. history to start learning**  | 80,000 frames        | 200,000 frames   | 200,000 frames   | 80,000 frames      | 200,000 frames   |
| **Target network update frequency** | 32,000 frames        | 40,000 frames    | 40,000 frames    | 32,000 frames      | 40,000 frames    |

## Visualization
We provide a [website](https://google.github.io/dopamine/baselines/atari/plots.html)
where you can quickly visualize the training runs for all our default agents.

The plots are rendered from a set of
[JSON files](https://github.com/google/dopamine/tree/master/baselines/atari/data)
which we compiled. These may prove useful in their own right to compare
against results obtained from other frameworks.


[dqn]: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
[c51]: https://arxiv.org/abs/1707.06887
[rainbow]: https://arxiv.org/abs/1710.02298
[iqn]: https://arxiv.org/abs/1806.06923

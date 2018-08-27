<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="rainbow_agent" />
<meta itemprop="path" content="stable" />
</div>

# Module: rainbow_agent

Compact implementation of a simplified Rainbow agent.

Specifically, we implement the following components from Rainbow:

*   n-step updates;
*   prioritized replay; and
*   distributional RL.

These three components were found to significantly impact the performance of the
Atari game-playing agent.

Furthermore, our implementation does away with some minor hyperparameter
choices. Specifically, we

*   keep the beta exponent fixed at beta=0.5, rather than increase it linearly;
*   remove the alpha parameter, which was set to alpha=0.5 throughout the paper.

Details in "Rainbow: Combining Improvements in Deep Reinforcement Learning" by
Hessel et al. (2018).

## Classes

[`class RainbowAgent`](./rainbow_agent/RainbowAgent.md): A compact
implementation of a simplified Rainbow agent.

## Functions

[`project_distribution(...)`](./rainbow_agent/project_distribution.md): Projects
a batch of (support, weights) onto target_support.

description: Compact implementation of a simplified Rainbow agent in Jax.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.jax.agents.rainbow.rainbow_agent" />
<meta itemprop="path" content="Stable" />
</div>

# Module: dopamine.jax.agents.rainbow.rainbow_agent

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/jax/agents/rainbow/rainbow_agent.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Compact implementation of a simplified Rainbow agent in Jax.

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

## Functions

[`project_distribution(...)`](../../../../dopamine/jax/agents/rainbow/rainbow_agent/project_distribution.md):
Projects a batch of (support, weights) onto target_support.

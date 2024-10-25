description: Compact implementation of the full Rainbow agent in JAX.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.jax.agents.full_rainbow.full_rainbow_agent" />
<meta itemprop="path" content="Stable" />
</div>

# Module: dopamine.jax.agents.full_rainbow.full_rainbow_agent

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/jax/agents/full_rainbow/full_rainbow_agent.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Compact implementation of the full Rainbow agent in JAX.


Specifically, we implement the following components from Rainbow:

  * n-step updates
  * prioritized replay
  * distributional RL
  * double_dqn
  * noisy
  * dueling

Details in "Rainbow: Combining Improvements in Deep Reinforcement Learning" by
Hessel et al. (2018).

## Classes

[`class JaxFullRainbowAgent`](../../../../dopamine/jax/agents/full_rainbow/full_rainbow_agent/JaxFullRainbowAgent.md): A compact implementation of the full Rainbow agent.


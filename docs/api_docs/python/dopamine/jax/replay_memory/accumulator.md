description: An accumulator used by the Jax replay buffer.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.jax.replay_memory.accumulator" />
<meta itemprop="path" content="Stable" />
</div>

# Module: dopamine.jax.replay_memory.accumulator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/jax/replay_memory/accumulator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



An accumulator used by the Jax replay buffer.



## Classes

[`class Accumulator`](../../../dopamine/jax/replay_memory/accumulator/Accumulator.md): Checkpointable protocol. Must implement to_state_dict, from_state_dict.

[`class TransitionAccumulator`](../../../dopamine/jax/replay_memory/accumulator/TransitionAccumulator.md): A transition accumulator used for preparing elements for replay.


description: A transition accumulator used for preparing elements for replay.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.jax.replay_memory.accumulator.TransitionAccumulator" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.jax.replay_memory.accumulator.TransitionAccumulator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/jax/replay_memory/accumulator.py#L42-L213">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A transition accumulator used for preparing elements for replay.

Inherits From: [`Accumulator`](../../../../dopamine/jax/replay_memory/accumulator/Accumulator.md), [`Checkpointable`](../../../../dopamine/jax/checkpointers/Checkpointable.md)

<!-- Placeholder for "Used in" -->

This class will consume raw transitions and prepare them for storing in the
replay buffer. Specifically, it will only return validly stacked frames, and
will only return valid n-step returns.
This enables us to guarantee that the replay buffer only contains valid
elements to sample.


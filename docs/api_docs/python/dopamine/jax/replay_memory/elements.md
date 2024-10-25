description: Various containers used by the replay buffer code.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.jax.replay_memory.elements" />
<meta itemprop="path" content="Stable" />
</div>

# Module: dopamine.jax.replay_memory.elements

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/jax/replay_memory/elements.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Various containers used by the replay buffer code.



## Classes

[`class ReplayElement`](../../../dopamine/jax/replay_memory/elements/ReplayElement.md): A single replay transition element supporting compression.

[`class ReplayElementProtocol`](../../../dopamine/jax/replay_memory/elements/ReplayElementProtocol.md): Base class for protocol classes.

[`class TransitionElement`](../../../dopamine/jax/replay_memory/elements/TransitionElement.md): TransitionElement(observation, action, reward, is_terminal, episode_end)

## Functions

[`compress(...)`](../../../dopamine/jax/replay_memory/elements/compress.md): Compress a numpy array using snappy.

[`uncompress(...)`](../../../dopamine/jax/replay_memory/elements/uncompress.md): Uncompress a numpy array that has been compressed via `compress`.


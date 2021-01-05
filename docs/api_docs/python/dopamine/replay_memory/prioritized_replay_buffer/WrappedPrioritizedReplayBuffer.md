description: Wrapper of OutOfGraphPrioritizedReplayBuffer with in-graph
sampling.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.replay_memory.prioritized_replay_buffer.WrappedPrioritizedReplayBuffer" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.replay_memory.prioritized_replay_buffer.WrappedPrioritizedReplayBuffer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/replay_memory/prioritized_replay_buffer.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Wrapper of OutOfGraphPrioritizedReplayBuffer with in-graph sampling.

Inherits From:
[`WrappedReplayBuffer`](../../../dopamine/replay_memory/circular_replay_buffer/WrappedReplayBuffer.md)

<!-- Placeholder for "Used in" -->

#### Usage:

*   To add a transition: Call the add function.

*   To sample a batch: Query any of the tensors in the transition dictionary.
    Every sess.run that requires any of these tensors will sample a new
    transition.

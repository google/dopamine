description: Wrapper of OutOfGraphReplayBuffer with an in graph sampling
mechanism.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.replay_memory.circular_replay_buffer.WrappedReplayBuffer" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.replay_memory.circular_replay_buffer.WrappedReplayBuffer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/replay_memory/circular_replay_buffer.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Wrapper of OutOfGraphReplayBuffer with an in graph sampling mechanism.

<!-- Placeholder for "Used in" -->

#### Usage:

To add a transition: call the add function.

To sample a batch: Construct operations that depend on any of the tensors is the
transition dictionary. Every sess.run that requires any of these tensors will
sample a new transition.

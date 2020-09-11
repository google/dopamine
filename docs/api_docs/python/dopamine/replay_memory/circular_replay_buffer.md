description: The standard DQN replay memory.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.replay_memory.circular_replay_buffer" />
<meta itemprop="path" content="Stable" />
</div>

# Module: dopamine.replay_memory.circular_replay_buffer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/replay_memory/circular_replay_buffer.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

The standard DQN replay memory.

This implementation is an out-of-graph replay memory + in-graph wrapper. It
supports vanilla n-step updates of the form typically found in the literature,
i.e. where rewards are accumulated for n steps and the intermediate trajectory
is not exposed to the agent. This does not allow, for example, performing
off-policy corrections.

## Classes

[`class OutOfGraphReplayBuffer`](../../dopamine/replay_memory/circular_replay_buffer/OutOfGraphReplayBuffer.md):
A simple out-of-graph Replay Buffer.

[`class WrappedReplayBuffer`](../../dopamine/replay_memory/circular_replay_buffer/WrappedReplayBuffer.md):
Wrapper of OutOfGraphReplayBuffer with an in graph sampling mechanism.

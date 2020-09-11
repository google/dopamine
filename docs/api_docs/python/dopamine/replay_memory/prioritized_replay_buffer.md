description: An implementation of Prioritized Experience Replay (PER).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.replay_memory.prioritized_replay_buffer" />
<meta itemprop="path" content="Stable" />
</div>

# Module: dopamine.replay_memory.prioritized_replay_buffer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/replay_memory/prioritized_replay_buffer.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

An implementation of Prioritized Experience Replay (PER).

This implementation is based on the paper "Prioritized Experience Replay" by Tom
Schaul et al. (2015). Many thanks to Tom Schaul, John Quan, and Matteo Hessel
for providing useful pointers on the algorithm and its implementation.

## Classes

[`class OutOfGraphPrioritizedReplayBuffer`](../../dopamine/replay_memory/prioritized_replay_buffer/OutOfGraphPrioritizedReplayBuffer.md):
An out-of-graph Replay Buffer for Prioritized Experience Replay.

[`class WrappedPrioritizedReplayBuffer`](../../dopamine/replay_memory/prioritized_replay_buffer/WrappedPrioritizedReplayBuffer.md):
Wrapper of OutOfGraphPrioritizedReplayBuffer with in-graph sampling.

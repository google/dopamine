<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="prioritized_replay_buffer" />
<meta itemprop="path" content="stable" />
</div>

# Module: prioritized_replay_buffer

An implementation of Prioritized Experience Replay (PER).

This implementation is based on the paper "Prioritized Experience Replay" by Tom
Schaul et al. (2015). Many thanks to Tom Schaul, John Quan, and Matteo Hessel
for providing useful pointers on the algorithm and its implementation.

## Classes

[`class OutOfGraphPrioritizedReplayBuffer`](./prioritized_replay_buffer/OutOfGraphPrioritizedReplayBuffer.md):
An out-of-graph Replay Buffer for Prioritized Experience Replay.

[`class WrappedPrioritizedReplayBuffer`](./prioritized_replay_buffer/WrappedPrioritizedReplayBuffer.md):
Wrapper of OutOfGraphPrioritizedReplayBuffer with in-graph sampling.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="circular_replay_buffer" />
<meta itemprop="path" content="stable" />
</div>

# Module: circular_replay_buffer

The standard DQN replay memory.

This implementation is an out-of-graph replay memory + in-graph wrapper. It
supports vanilla n-step updates of the form typically found in the literature,
i.e. where rewards are accumulated for n steps and the intermediate trajectory
is not exposed to the agent. This does not allow, for example, performing
off-policy corrections.

## Classes

[`class OutOfGraphReplayBuffer`](./circular_replay_buffer/OutOfGraphReplayBuffer.md):
A simple out-of-graph Replay Buffer.

[`class WrappedReplayBuffer`](./circular_replay_buffer/WrappedReplayBuffer.md):
Wrapper of OutOfGraphReplayBuffer with an in graph sampling mechanism.

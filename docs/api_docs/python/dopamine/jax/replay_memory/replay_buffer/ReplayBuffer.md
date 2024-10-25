description: A Jax re-implementation of the Dopamine Replay Buffer.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.jax.replay_memory.replay_buffer.ReplayBuffer" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.jax.replay_memory.replay_buffer.ReplayBuffer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/jax/replay_memory/replay_buffer.py#L36-L268">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A Jax re-implementation of the Dopamine Replay Buffer.

Inherits From: [`Checkpointable`](../../../../dopamine/jax/checkpointers/Checkpointable.md)

<!-- Placeholder for "Used in" -->

Stores transitions, state, action, reward, next_state, terminal (and any
extra contents specified) in a circular buffer and provides a uniform
sampling mechanism.

The main changes from the original Dopamine implementation are:
- The original Dopamine replay buffer stored the raw observations and stacked
  them upon sampling. Although space efficient, this is time inefficient. We
  use the same compression mechanism used by dqn_zoo to store only stacked
  states in a space efficient manner.
- Similarly, n-step transitions were computed at sample time. We only store
  n-step returns in the buffer by using an Accumulator (which is also used for
  stacking).
- The above two points allow us to sample uniformly from the FIFO queue,
  without needing to determine invalid ranges (as in the original replay
  buffer). The computation of invalid ranges was inefficient, as it required
  verifying each sampled index, and potentially resampling them if they fell
  in an invalid range. This computation scales linearly with the batch size,
  and is thus quite inefficient.
- The original Dopamine replay buffer maintained a static array and performed
  modulo arithmetic when adding/sampling. This is unnecessarily complicated
  and we can achieve the same result by maintaining a FIFO queue (using an
  OrderedDict structure) containing only valid transitions.


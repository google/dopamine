description: A simple out-of-graph Replay Buffer.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.tf.replay_memory.circular_replay_buffer.OutOfGraphReplayBuffer" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.tf.replay_memory.circular_replay_buffer.OutOfGraphReplayBuffer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/tf/replay_memory/circular_replay_buffer.py#L83-L815">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A simple out-of-graph Replay Buffer.

<!-- Placeholder for "Used in" -->

Stores transitions, state, action, reward, next_state, terminal (and any
extra contents specified) in a circular buffer and provides a uniform
transition sampling function.

When the states consist of stacks of observations storing the states is
inefficient. This class writes observations and constructs the stacked states
at sample time.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`add_count`<a id="add_count"></a>
</td>
<td>
int, counter of how many transitions have been added (including
the blank ones at the beginning of an episode).
</td>
</tr><tr>
<td>
`invalid_range`<a id="invalid_range"></a>
</td>
<td>
np.array, an array with the indices of cursor-related invalid
transitions
</td>
</tr><tr>
<td>
`episode_end_indices`<a id="episode_end_indices"></a>
</td>
<td>
set[int], a set of indices corresponding to the end of
an episode.
</td>
</tr>
</table>




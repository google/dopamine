description: Wrapper of OutOfGraphPrioritizedReplayBuffer with in-graph sampling.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.tf.replay_memory.prioritized_replay_buffer.WrappedPrioritizedReplayBuffer" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.tf.replay_memory.prioritized_replay_buffer.WrappedPrioritizedReplayBuffer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/tf/replay_memory/prioritized_replay_buffer.py#L268-L399">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Wrapper of OutOfGraphPrioritizedReplayBuffer with in-graph sampling.

Inherits From: [`WrappedReplayBuffer`](../../../../dopamine/tf/replay_memory/circular_replay_buffer/WrappedReplayBuffer.md)

<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Usage</h2></th></tr>
<tr class="alt">
<td colspan="2">
* To add a transition:  Call the add function.

* To sample a batch:  Query any of the tensors in the transition dictionary.
                      Every sess.run that requires any of these tensors will
                      sample a new transition.
</td>
</tr>

</table>




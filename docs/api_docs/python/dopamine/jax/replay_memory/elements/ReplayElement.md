description: A single replay transition element supporting compression.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.jax.replay_memory.elements.ReplayElement" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.jax.replay_memory.elements.ReplayElement

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/jax/replay_memory/elements.py#L96-L138">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A single replay transition element supporting compression.

Inherits From: [`ReplayElementProtocol`](../../../../dopamine/jax/replay_memory/elements/ReplayElementProtocol.md)

<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`state`<a id="state"></a>
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`action`<a id="action"></a>
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`reward`<a id="reward"></a>
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`next_state`<a id="next_state"></a>
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`is_terminal`<a id="is_terminal"></a>
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`episode_end`<a id="episode_end"></a>
</td>
<td>
Dataclass field
</td>
</tr>
</table>




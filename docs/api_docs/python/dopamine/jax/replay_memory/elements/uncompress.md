description: Uncompress a numpy array that has been compressed via compress.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.jax.replay_memory.elements.uncompress" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.jax.replay_memory.elements.uncompress

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/jax/replay_memory/elements.py#L76-L93">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Uncompress a numpy array that has been compressed via `compress`.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dopamine.jax.replay_memory.elements.uncompress(
    compressed: npt.NDArray
) -> npt.NDArray
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`compressed`<a id="compressed"></a>
</td>
<td>
npt.NDArray, numpy structured array with data, shape, and dtype.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Uncompressed npt.NDArray
</td>
</tr>

</table>


description: Compress a numpy array using snappy.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.jax.replay_memory.elements.compress" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.jax.replay_memory.elements.compress

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/jax/replay_memory/elements.py#L49-L73">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Compress a numpy array using snappy.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dopamine.jax.replay_memory.elements.compress(
    buffer: npt.NDArray
) -> npt.NDArray
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`buffer`<a id="buffer"></a>
</td>
<td>
npt.NDArray, buffer to compress.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Numpy structured array consisting of the following fields:
data: compressed data in bytes
shape: the shape of the uncompressed array
dtype: a string representation of the dtype
</td>
</tr>

</table>


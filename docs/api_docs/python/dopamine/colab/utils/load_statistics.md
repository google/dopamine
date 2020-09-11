description: Reads in a statistics object from log_path.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.colab.utils.load_statistics" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.colab.utils.load_statistics

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/colab/utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Reads in a statistics object from log_path.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dopamine.colab.utils.load_statistics(
    log_path, iteration_number=None, verbose=True
)
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`log_path`
</td>
<td>
string, provides the full path to the training/eval statistics.
</td>
</tr><tr>
<td>
`iteration_number`
</td>
<td>
The iteration number of the statistics object we want
to read. If set to None, load the latest version.
</td>
</tr><tr>
<td>
`verbose`
</td>
<td>
Whether to output information about the load procedure.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`data`
</td>
<td>
The requested statistics object.
</td>
</tr><tr>
<td>
`iteration`
</td>
<td>
The corresponding iteration number.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`Exception`
</td>
<td>
if data is not present.
</td>
</tr>
</table>

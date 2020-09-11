description: Reads in the baseline experimental data from a specified base
directory.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.colab.utils.load_baselines" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.colab.utils.load_baselines

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/colab/utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Reads in the baseline experimental data from a specified base directory.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dopamine.colab.utils.load_baselines(
    base_dir, verbose=False
)
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`base_dir`
</td>
<td>
string, base directory where to read data from.
</td>
</tr><tr>
<td>
`verbose`
</td>
<td>
bool, whether to print warning messages.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A dict containing pandas DataFrames for all available agents and games.
</td>
</tr>

</table>

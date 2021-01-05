description: Processes log data into a per-iteration summary.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.colab.utils.summarize_data" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.colab.utils.summarize_data

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/colab/utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Processes log data into a per-iteration summary.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dopamine.colab.utils.summarize_data(
    data, summary_keys
)
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`data`
</td>
<td>
Dictionary loaded by load_statistics describing the data. This
dictionary has keys iteration_0, iteration_1, ... describing per-iteration
data.
</td>
</tr><tr>
<td>
`summary_keys`
</td>
<td>
List of per-iteration data to be summarized.
</td>
</tr>
</table>

#### Example:

data = load_statistics(...) summarize_data(data, ['train_episode_returns',
'eval_episode_returns'])

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A dictionary mapping each key in returns_keys to a per-iteration summary.
</td>
</tr>

</table>

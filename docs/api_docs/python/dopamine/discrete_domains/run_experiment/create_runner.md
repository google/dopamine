description: Creates an experiment Runner.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.discrete_domains.run_experiment.create_runner" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.discrete_domains.run_experiment.create_runner

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/discrete_domains/run_experiment.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Creates an experiment Runner.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dopamine.discrete_domains.run_experiment.create_runner(
    base_dir, schedule='continuous_train_and_eval'
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
str, base directory for hosting all subdirectories.
</td>
</tr><tr>
<td>
`schedule`
</td>
<td>
string, which type of Runner to use.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`runner`
</td>
<td>
A `Runner` like object.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
When an unknown schedule is encountered.
</td>
</tr>
</table>

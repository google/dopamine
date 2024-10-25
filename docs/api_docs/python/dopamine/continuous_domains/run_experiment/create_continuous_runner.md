description: Creates an experiment Runner.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.continuous_domains.run_experiment.create_continuous_runner" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.continuous_domains.run_experiment.create_continuous_runner

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/continuous_domains/run_experiment.py#L116-L138">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Creates an experiment Runner.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dopamine.continuous_domains.run_experiment.create_continuous_runner(
    base_dir, schedule=&#x27;continuous_train_and_eval&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`base_dir`<a id="base_dir"></a>
</td>
<td>
str, base directory for hosting all subdirectories.
</td>
</tr><tr>
<td>
`schedule`<a id="schedule"></a>
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
`runner`<a id="runner"></a>
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
`ValueError`<a id="ValueError"></a>
</td>
<td>
When an unknown schedule is encountered.
</td>
</tr>
</table>


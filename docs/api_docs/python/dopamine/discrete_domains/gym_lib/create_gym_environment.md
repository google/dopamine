description: Wraps a Gym environment with some basic preprocessing.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.discrete_domains.gym_lib.create_gym_environment" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.discrete_domains.gym_lib.create_gym_environment

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/discrete_domains/gym_lib.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Wraps a Gym environment with some basic preprocessing.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dopamine.discrete_domains.gym_lib.create_gym_environment(
    environment_name=None, version='v0'
)
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`environment_name`
</td>
<td>
str, the name of the environment to run.
</td>
</tr><tr>
<td>
`version`
</td>
<td>
str, version of the environment to run.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Gym environment with some standard preprocessing.
</td>
</tr>

</table>

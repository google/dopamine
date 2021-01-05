description: Creates an agent.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.discrete_domains.run_experiment.create_agent" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.discrete_domains.run_experiment.create_agent

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/discrete_domains/run_experiment.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Creates an agent.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dopamine.discrete_domains.run_experiment.create_agent(
    sess, environment, agent_name=None, summary_writer=None, debug_mode=False
)
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`sess`
</td>
<td>
A `tf.compat.v1.Session` object for running associated ops.
</td>
</tr><tr>
<td>
`environment`
</td>
<td>
A gym environment (e.g. Atari 2600).
</td>
</tr><tr>
<td>
`agent_name`
</td>
<td>
str, name of the agent to create.
</td>
</tr><tr>
<td>
`summary_writer`
</td>
<td>
A Tensorflow summary writer to pass to the agent
for in-agent training statistics in Tensorboard.
</td>
</tr><tr>
<td>
`debug_mode`
</td>
<td>
bool, whether to output Tensorboard summaries. If set to true,
the agent will output in-episode statistics to Tensorboard. Disabled by
default as this results in slower training.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`agent`
</td>
<td>
An RL agent.
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
If `agent_name` is not in supported list.
</td>
</tr>
</table>

description: Creates an agent.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.continuous_domains.run_experiment.create_continuous_agent" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.continuous_domains.run_experiment.create_continuous_agent

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/continuous_domains/run_experiment.py#L47-L113">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Creates an agent.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dopamine.continuous_domains.run_experiment.create_continuous_agent(
    environment: <a href="../../../dopamine/discrete_domains/gym_lib/GymPreprocessing.md"><code>dopamine.discrete_domains.gym_lib.GymPreprocessing</code></a>,
    agent_name: str,
    summary_writer: Optional[tensorboard.SummaryWriter] = None
) -> <a href="../../../dopamine/jax/agents/dqn/dqn_agent/JaxDQNAgent.md"><code>dopamine.jax.agents.dqn.dqn_agent.JaxDQNAgent</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`environment`<a id="environment"></a>
</td>
<td>
 A gym environment.
</td>
</tr><tr>
<td>
`agent_name`<a id="agent_name"></a>
</td>
<td>
str, name of the agent to create.
</td>
</tr><tr>
<td>
`summary_writer`<a id="summary_writer"></a>
</td>
<td>
A Tensorflow summary writer to pass to the agent for
in-agent training statistics in Tensorboard.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
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
`ValueError`<a id="ValueError"></a>
</td>
<td>
If `agent_name` is not in supported list.
</td>
</tr>
</table>


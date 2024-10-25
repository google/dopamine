description: Build the deep network used to compute the agent's Q-value distributions.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.discrete_domains.legacy_networks.acrobot_rainbow_network" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.discrete_domains.legacy_networks.acrobot_rainbow_network

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/discrete_domains/legacy_networks.py#L226-L244">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Build the deep network used to compute the agent's Q-value distributions.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dopamine.discrete_domains.legacy_networks.acrobot_rainbow_network(
    num_actions, num_atoms, support, network_type, state
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`num_actions`<a id="num_actions"></a>
</td>
<td>
int, number of actions.
</td>
</tr><tr>
<td>
`num_atoms`<a id="num_atoms"></a>
</td>
<td>
int, the number of buckets of the value function distribution.
</td>
</tr><tr>
<td>
`support`<a id="support"></a>
</td>
<td>
tf.linspace, the support of the Q-value distribution.
</td>
</tr><tr>
<td>
`network_type`<a id="network_type"></a>
</td>
<td>
`namedtuple`, collection of expected values to return.
</td>
</tr><tr>
<td>
`state`<a id="state"></a>
</td>
<td>
`tf.Tensor`, contains the agent's current state.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`net`<a id="net"></a>
</td>
<td>
_network_type object containing the tensors output by the network.
</td>
</tr>
</table>


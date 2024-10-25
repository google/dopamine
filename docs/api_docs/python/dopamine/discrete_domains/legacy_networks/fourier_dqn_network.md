description: Builds the function approximator used to compute the agent's Q-values.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.discrete_domains.legacy_networks.fourier_dqn_network" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.discrete_domains.legacy_networks.fourier_dqn_network

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/discrete_domains/legacy_networks.py#L125-L146">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Builds the function approximator used to compute the agent's Q-values.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dopamine.discrete_domains.legacy_networks.fourier_dqn_network(
    min_vals, max_vals, num_actions, state, fourier_basis_order=3
)
</code></pre>



<!-- Placeholder for "Used in" -->

It uses FourierBasis features and a linear layer.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`min_vals`<a id="min_vals"></a>
</td>
<td>
float, minimum attainable values (must be same shape as `state`).
</td>
</tr><tr>
<td>
`max_vals`<a id="max_vals"></a>
</td>
<td>
float, maximum attainable values (must be same shape as `state`).
</td>
</tr><tr>
<td>
`num_actions`<a id="num_actions"></a>
</td>
<td>
int, number of actions.
</td>
</tr><tr>
<td>
`state`<a id="state"></a>
</td>
<td>
`tf.Tensor`, contains the agent's current state.
</td>
</tr><tr>
<td>
`fourier_basis_order`<a id="fourier_basis_order"></a>
</td>
<td>
int, order of the Fourier basis functions.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The Q-values for DQN-style agents or logits for Rainbow-style agents.
</td>
</tr>

</table>


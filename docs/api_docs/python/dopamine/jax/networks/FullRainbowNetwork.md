description: Jax Rainbow network for Full Rainbow.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.jax.networks.FullRainbowNetwork" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.jax.networks.FullRainbowNetwork

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/jax/networks.py#L543-L605">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Jax Rainbow network for Full Rainbow.

<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`num_actions`<a id="num_actions"></a>
</td>
<td>
int, number of actions the agent can take at any state.
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
`noisy`<a id="noisy"></a>
</td>
<td>
bool, Whether to use noisy networks.
</td>
</tr><tr>
<td>
`dueling`<a id="dueling"></a>
</td>
<td>
bool, Whether to use dueling network architecture.
</td>
</tr><tr>
<td>
`distributional`<a id="distributional"></a>
</td>
<td>
bool, whether to use distributional RL.
</td>
</tr><tr>
<td>
`inputs_preprocessed`<a id="inputs_preprocessed"></a>
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`parent`<a id="parent"></a>
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
Dataclass field
</td>
</tr>
</table>




description: Projects a batch of (support, weights) onto target_support.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.jax.agents.rainbow.rainbow_agent.project_distribution" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.jax.agents.rainbow.rainbow_agent.project_distribution

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/jax/agents/rainbow/rainbow_agent.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Projects a batch of (support, weights) onto target_support.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dopamine.jax.agents.rainbow.rainbow_agent.project_distribution(
    supports, weights, target_support
)
</code></pre>

<!-- Placeholder for "Used in" -->

Based on equation (7) in (Bellemare et al., 2017):
https://arxiv.org/abs/1707.06887 In the rest of the comments we will refer to
this equation simply as Eq7.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`supports`
</td>
<td>
Jax array of shape (num_dims) defining supports for
the distribution.
</td>
</tr><tr>
<td>
`weights`
</td>
<td>
Jax array of shape (num_dims) defining weights on the
original support points. Although for the CategoricalDQN agent these
weights are probabilities, it is not required that they are.
</td>
</tr><tr>
<td>
`target_support`
</td>
<td>
Jax array of shape (num_dims) defining support of the
projected distribution. The values must be monotonically increasing. Vmin
and Vmax will be inferred from the first and last elements of this Jax
array, respectively. The values in this Jax array must be equally spaced.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Jax array of shape (num_dims) with the projection of a batch
of (support, weights) onto target_support.
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
If target_support has no dimensions, or if shapes of supports,
weights, and target_support are incompatible.
</td>
</tr>
</table>

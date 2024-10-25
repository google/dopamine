description: Implementation of the Huber loss with threshold delta.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.jax.losses.huber_loss" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.jax.losses.huber_loss

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/jax/losses.py#L19-L37">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Implementation of the Huber loss with threshold delta.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dopamine.jax.losses.huber_loss(
    targets: jnp.ndarray, predictions: jnp.ndarray, delta: float = 1.0
) -> jnp.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->

Let `x = |targets - predictions|`, the Huber loss is defined as:
`0.5 * x^2` if `x <= delta`
`0.5 * delta^2 + delta * (x - delta)` otherwise.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`targets`<a id="targets"></a>
</td>
<td>
Target values.
</td>
</tr><tr>
<td>
`predictions`<a id="predictions"></a>
</td>
<td>
Prediction values.
</td>
</tr><tr>
<td>
`delta`<a id="delta"></a>
</td>
<td>
Threshold.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Huber loss.
</td>
</tr>

</table>


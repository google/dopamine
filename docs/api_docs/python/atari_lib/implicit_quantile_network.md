<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="atari_lib.implicit_quantile_network" />
<meta itemprop="path" content="Stable" />
</div>

# atari_lib.implicit_quantile_network

```python
atari_lib.implicit_quantile_network(
    num_actions,
    quantile_embedding_dim,
    network_type,
    state,
    num_quantiles
)
```

The Implicit Quantile ConvNet.

#### Args:

*   <b>`num_actions`</b>: int, number of actions.
*   <b>`quantile_embedding_dim`</b>: int, embedding dimension for the quantile
    input.
*   <b>`network_type`</b>: namedtuple, collection of expected values to return.
*   <b>`state`</b>: `tf.Tensor`, contains the agent's current state.
*   <b>`num_quantiles`</b>: int, number of quantile inputs.

#### Returns:

*   <b>`net`</b>: _network_type object containing the tensors output by the
    network.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="atari_lib.rainbow_network" />
<meta itemprop="path" content="Stable" />
</div>

# atari_lib.rainbow_network

```python
atari_lib.rainbow_network(
    num_actions,
    num_atoms,
    support,
    network_type,
    state
)
```

The convolutional network used to compute agent's Q-value distributions.

#### Args:

*   <b>`num_actions`</b>: int, number of actions.
*   <b>`num_atoms`</b>: int, the number of buckets of the value function
    distribution.
*   <b>`support`</b>: tf.linspace, the support of the Q-value distribution.
*   <b>`network_type`</b>: namedtuple, collection of expected values to return.
*   <b>`state`</b>: `tf.Tensor`, contains the agent's current state.

#### Returns:

*   <b>`net`</b>: _network_type object containing the tensors output by the
    network.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gym_lib.acrobot_rainbow_network" />
<meta itemprop="path" content="Stable" />
</div>

# gym_lib.acrobot_rainbow_network

```python
gym_lib.acrobot_rainbow_network(
    *args,
    **kwargs
)
```

Build the deep network used to compute the agent's Q-value distributions.

#### Args:

*   <b>`num_actions`</b>: int, number of actions.
*   <b>`num_atoms`</b>: int, the number of buckets of the value function
    distribution.
*   <b>`support`</b>: tf.linspace, the support of the Q-value distribution.
*   <b>`network_type`</b>: `namedtuple`, collection of expected values to
    return.
*   <b>`state`</b>: `tf.Tensor`, contains the agent's current state.

#### Returns:

*   <b>`net`</b>: _network_type object containing the tensors output by the
    network.

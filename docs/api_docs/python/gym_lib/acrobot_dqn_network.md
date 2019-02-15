<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="gym_lib.acrobot_dqn_network" />
<meta itemprop="path" content="Stable" />
</div>

# gym_lib.acrobot_dqn_network

```python
gym_lib.acrobot_dqn_network(
    *args,
    **kwargs
)
```

Builds the deep network used to compute the agent's Q-values.

It rescales the input features to a range that yields improved performance.

#### Args:

*   <b>`num_actions`</b>: int, number of actions.
*   <b>`network_type`</b>: namedtuple, collection of expected values to return.
*   <b>`state`</b>: `tf.Tensor`, contains the agent's current state.

#### Returns:

*   <b>`net`</b>: _network_type object containing the tensors output by the
    network.

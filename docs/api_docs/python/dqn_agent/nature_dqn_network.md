<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dqn_agent.nature_dqn_network" />
<meta itemprop="path" content="Stable" />
</div>

# dqn_agent.nature_dqn_network

```python
dqn_agent.nature_dqn_network(
    num_actions,
    network_type,
    state
)
```

The convolutional network used to compute the agent's Q-values.

#### Args:

*   <b>`num_actions`</b>: int, number of actions.
*   <b>`network_type`</b>: namedtuple, collection of expected values to return.
*   <b>`state`</b>: `tf.Tensor`, contains the agent's current state.

#### Returns:

*   <b>`net`</b>: _network_type object containing the tensors output by the
    network.

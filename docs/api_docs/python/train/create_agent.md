<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="train.create_agent" />
<meta itemprop="path" content="stable" />
</div>

# train.create_agent

```python
train.create_agent(
    sess,
    environment
)
```

Creates a DQN agent.

#### Args:

*   <b>`sess`</b>: A `tf.compat.v1.Session` object for running associated ops.
*   <b>`environment`</b>: An Atari 2600 Gym environment.

#### Returns:

*   <b>`agent`</b>: An RL agent.

#### Raises:

*   <b>`ValueError`</b>: If `agent_name` is not in supported list.

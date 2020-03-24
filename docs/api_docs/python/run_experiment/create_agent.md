<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="run_experiment.create_agent" />
<meta itemprop="path" content="Stable" />
</div>

# run_experiment.create_agent

```python
run_experiment.create_agent(
    *args,
    **kwargs
)
```

Creates an agent.

#### Args:

*   <b>`sess`</b>: A `tf.compat.v1.Session` object for running associated ops.
*   <b>`environment`</b>: An Atari 2600 Gym environment.
*   <b>`agent_name`</b>: str, name of the agent to create.
*   <b>`summary_writer`</b>: A Tensorflow summary writer to pass to the agent
    for in-agent training statistics in Tensorboard.
*   <b>`debug_mode`</b>: bool, whether to output Tensorboard summaries. If set
    to true, the agent will output in-episode statistics to Tensorboard.
    Disabled by default as this results in slower training.

#### Returns:

*   <b>`agent`</b>: An RL agent.

#### Raises:

*   <b>`ValueError`</b>: If `agent_name` is not in supported list.

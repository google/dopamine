<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="train.launch_experiment" />
<meta itemprop="path" content="stable" />
</div>

# train.launch_experiment

```python
train.launch_experiment(
    create_runner_fn,
    create_agent_fn
)
```

Launches the experiment.

#### Args:

*   <b>`create_runner_fn`</b>: A function that takes as args a base directory
    and a function for creating an agent and returns a `Runner`-like object.
*   <b>`create_agent_fn`</b>: A function that takes as args a Tensorflow session
    and an Atari 2600 Gym environment, and returns an agent.

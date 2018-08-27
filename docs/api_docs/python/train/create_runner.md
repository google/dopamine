<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="train.create_runner" />
<meta itemprop="path" content="stable" />
</div>

# train.create_runner

```python
train.create_runner(
    base_dir,
    create_agent_fn
)
```

Creates an experiment Runner.

#### Args:

*   <b>`base_dir`</b>: str, base directory for hosting all subdirectories.
*   <b>`create_agent_fn`</b>: A function that takes as args a Tensorflow session
    and an Atari 2600 Gym environment, and returns an agent.

#### Returns:

*   <b>`runner`</b>: A
    <a href="../run_experiment/Runner.md"><code>run_experiment.Runner</code></a>
    like object.

#### Raises:

*   <b>`ValueError`</b>: When an unknown schedule is encountered.

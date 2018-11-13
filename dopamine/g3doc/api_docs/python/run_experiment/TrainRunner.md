<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="run_experiment.TrainRunner" />
<meta itemprop="path" content="stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="run_experiment"/>
</div>

# run_experiment.TrainRunner

## Class `TrainRunner`

Inherits From: [`Runner`](../run_experiment/Runner.md)

Object that handles running Atari 2600 experiments.

The `TrainRunner` differs from the base `Runner` class in that it does not the
evaluation phase. Checkpointing and logging for the train phase are preserved as
before.

## Methods

<h3 id="__init__"><code>__init__</code></h3>

```python
__init__(
    *args,
    **kwargs
)
```

Initialize the TrainRunner object in charge of running a full experiment.

#### Args:

*   <b>`base_dir`</b>: str, the base directory to host all required
    sub-directories.
*   <b>`create_agent_fn`</b>: A function that takes as args a Tensorflow session
    and an Atari 2600 Gym environment, and returns an agent.

<h3 id="run_experiment"><code>run_experiment</code></h3>

```python
run_experiment()
```

Runs a full experiment, spread over multiple iterations.

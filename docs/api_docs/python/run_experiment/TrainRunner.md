<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="run_experiment.TrainRunner" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="run_experiment"/>
</div>

# run_experiment.TrainRunner

## Class `TrainRunner`

Inherits From: [`Runner`](../run_experiment/Runner.md)

Object that handles running experiments.

The `TrainRunner` differs from the base `Runner` class in that it does not the
evaluation phase. Checkpointing and logging for the train phase are preserved as
before.

<h2 id="__init__"><code>__init__</code></h2>

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
    and an environment, and returns an agent.
*   <b>`create_environment_fn`</b>: A function which receives a problem name and
    creates a Gym environment for that problem (e.g. an Atari 2600 game).

## Methods

<h3 id="run_experiment"><code>run_experiment</code></h3>

```python
run_experiment()
```

Runs a full experiment, spread over multiple iterations.

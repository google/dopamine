<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="run_experiment.Runner" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="run_experiment"/>
</div>

# run_experiment.Runner

## Class `Runner`

Object that handles running Dopamine experiments.

Here we use the term 'experiment' to mean simulating interactions between the
agent and the environment and reporting some statistics pertaining to these
interactions.

A simple scenario to train a DQN agent is as follows:

```python
import dopamine.discrete_domains.atari_lib
base_dir = '/tmp/simple_example'
def create_agent(sess, environment):
  return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n)
runner = Runner(base_dir, create_agent, atari_lib.create_atari_environment)
runner.run()
```

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(
    *args,
    **kwargs
)
```

Initialize the Runner object in charge of running a full experiment.

#### Args:

*   <b>`base_dir`</b>: str, the base directory to host all required
    sub-directories.
*   <b>`create_agent_fn`</b>: A function that takes as args a Tensorflow session
    and an environment, and returns an agent.
*   <b>`create_environment_fn`</b>: A function which receives a problem name and
    creates a Gym environment for that problem (e.g. an Atari 2600 game).
*   <b>`checkpoint_file_prefix`</b>: str, the prefix to use for checkpoint
    files.
*   <b>`logging_file_prefix`</b>: str, prefix to use for the log files.
*   <b>`log_every_n`</b>: int, the frequency for writing logs.
*   <b>`num_iterations`</b>: int, the iteration number threshold (must be
    greater than start_iteration).
*   <b>`training_steps`</b>: int, the number of training steps to perform.
*   <b>`evaluation_steps`</b>: int, the number of evaluation steps to perform.
*   <b>`max_steps_per_episode`</b>: int, maximum number of steps after which an
    episode terminates.

This constructor will take the following actions: - Initialize an environment. -
Initialize a `tf.compat.v1.Session`. - Initialize a logger. - Initialize an agent. -
Reload from the latest checkpoint, if available, and initialize the Checkpointer
object.

## Methods

<h3 id="run_experiment"><code>run_experiment</code></h3>

```python
run_experiment()
```

Runs a full experiment, spread over multiple iterations.

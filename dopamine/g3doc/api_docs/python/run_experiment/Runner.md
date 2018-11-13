<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="run_experiment.Runner" />
<meta itemprop="path" content="stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="run_experiment"/>
</div>

# run_experiment.Runner

## Class `Runner`

Object that handles running Atari 2600 experiments.

Here we use the term 'experiment' to mean simulating interactions between the
agent and the environment and reporting some statistics pertaining to these
interactions.

A simple scenario to train a DQN agent is as follows:

```python
base_dir = '/tmp/simple_example'
def create_agent(sess, environment):
  return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n)
runner = Runner(base_dir, create_agent, game_name='Pong')
runner.run()
```

## Methods

<h3 id="__init__"><code>__init__</code></h3>

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
    and an Atari 2600 Gym environment, and returns an agent.
*   <b>`create_environment_fn`</b>: A function which receives a game name and
    creates an Atari 2600 Gym environment.
*   <b>`game_name`</b>: str, name of the Atari 2600 domain to run (required).
*   <b>`sticky_actions`</b>: bool, whether to enable sticky actions in the
    environment.
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
Initialize a `tf.Session`. - Initialize a logger. - Initialize an agent. -
Reload from the latest checkpoint, if available, and initialize the Checkpointer
object.

<h3 id="run_experiment"><code>run_experiment</code></h3>

```python
run_experiment()
```

Runs a full experiment, spread over multiple iterations.

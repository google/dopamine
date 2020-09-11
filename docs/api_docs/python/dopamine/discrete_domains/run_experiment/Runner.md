description: Object that handles running Dopamine experiments.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.discrete_domains.run_experiment.Runner" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.discrete_domains.run_experiment.Runner

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/discrete_domains/run_experiment.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Object that handles running Dopamine experiments.

<!-- Placeholder for "Used in" -->

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

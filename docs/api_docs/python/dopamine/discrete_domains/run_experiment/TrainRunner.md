description: Object that handles running experiments.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.discrete_domains.run_experiment.TrainRunner" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.discrete_domains.run_experiment.TrainRunner

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/discrete_domains/run_experiment.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Object that handles running experiments.

Inherits From:
[`Runner`](../../../dopamine/discrete_domains/run_experiment/Runner.md)

<!-- Placeholder for "Used in" -->

The `TrainRunner` differs from the base `Runner` class in that it does not the
evaluation phase. Checkpointing and logging for the train phase are preserved as
before.

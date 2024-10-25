description: Base class for metric collectors.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.metrics.collector" />
<meta itemprop="path" content="Stable" />
</div>

# Module: dopamine.metrics.collector

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/metrics/collector.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Base class for metric collectors.


Each Collector should subclass this base class, as the CollectorDispatcher
object expects objects of type Collector.

The methods to implement are:
  - `get_name`: a unique identifier for subdirectory creation.
  - `pre_training`: called once before training begins.
  - `step`: called once for each training step. The parameter is an object of
    type `StatisticsInstance` which contains the statistics of the current
    training step.
  - `end_training`: called once at the end of training, and passes in a
    `StatisticsInstance` containing the statistics of the latest training step.

## Classes

[`class Collector`](../../dopamine/metrics/collector/Collector.md): Abstract class for defining metric collectors.


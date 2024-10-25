description: Class that runs a list of Collectors for metrics reporting.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.metrics.collector_dispatcher" />
<meta itemprop="path" content="Stable" />
</div>

# Module: dopamine.metrics.collector_dispatcher

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/metrics/collector_dispatcher.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Class that runs a list of Collectors for metrics reporting.


This class is what should be called from the main binary and will call each of
the specified collectors for metrics reporting.

Each metric collector can be further configured via gin bindings. The
constructor for each desired collector should be passed in as a list when
creating this object. All of the collectors are expected to be subclasses of the
`Collector` base class (defined in `collector.py`).

#### Example configuration:


```
metrics = CollectorDispatcher(base_dir, num_actions, list_of_constructors)
metrics.pre_training()
for i in range(training_steps):
  ...
  metrics.step(statistics)
metrics.end_training(statistics)
```

The statistics are passed in as a dict that contains
and contains the raw performance statistics for the current iteration. All
processing (such as averaging) will be handled by each of the individual
collectors.

## Classes

[`class CollectorDispatcher`](../../dopamine/metrics/collector_dispatcher/CollectorDispatcher.md): Class for collecting and reporting Dopamine metrics.

## Functions

[`add_collector(...)`](../../dopamine/metrics/collector_dispatcher/add_collector.md)


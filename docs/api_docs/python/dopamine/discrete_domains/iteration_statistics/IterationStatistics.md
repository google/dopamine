description: A class for storing iteration-specific metrics.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.discrete_domains.iteration_statistics.IterationStatistics" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.discrete_domains.iteration_statistics.IterationStatistics

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/discrete_domains/iteration_statistics.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

A class for storing iteration-specific metrics.

<!-- Placeholder for "Used in" -->

The internal format is as follows: we maintain a mapping from keys to lists.
Each list contains all the values corresponding to the given key.

For example, self.data_lists['train_episode_returns'] might contain the
per-episode returns achieved during this iteration.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`data_lists`
</td>
<td>
dict mapping each metric_name (str) to a list of said metric
across episodes.
</td>
</tr>
</table>

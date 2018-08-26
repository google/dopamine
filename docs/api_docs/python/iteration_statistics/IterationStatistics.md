<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="iteration_statistics.IterationStatistics" />
<meta itemprop="path" content="stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="append"/>
</div>

# iteration_statistics.IterationStatistics

## Class `IterationStatistics`

A class for storing iteration-specific metrics.

The internal format is as follows: we maintain a mapping from keys to lists.
Each list contains all the values corresponding to the given key.

For example, self.data_lists['train_episode_returns'] might contain the
per-episode returns achieved during this iteration.

#### Attributes:

*   <b>`data_lists`</b>: dict mapping each metric_name (str) to a list of said
    metric across episodes.

## Methods

<h3 id="__init__"><code>__init__</code></h3>

```python
__init__()
```

<h3 id="append"><code>append</code></h3>

```python
append(data_pairs)
```

Add the given values to their corresponding key-indexed lists.

#### Args:

*   <b>`data_pairs`</b>: A dictionary of key-value pairs to be recorded.

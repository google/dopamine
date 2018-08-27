<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="utils.summarize_data" />
<meta itemprop="path" content="stable" />
</div>

# utils.summarize_data

```python
utils.summarize_data(
    data,
    summary_keys
)
```

Processes log data into a per-iteration summary.

#### Args:

*   <b>`data`</b>: Dictionary loaded by load_statistics describing the data.
    This dictionary has keys iteration_0, iteration_1, ... describing
    per-iteration data.
*   <b>`summary_keys`</b>: List of per-iteration data to be summarized.

Example: data = load_statistics(...) get_iteration_summmary(data,
['train_episode_returns', 'eval_episode_returns'])

#### Returns:

A dictionary mapping each key in returns_keys to a per-iteration summary.

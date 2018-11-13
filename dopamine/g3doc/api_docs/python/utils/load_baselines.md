<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="utils.load_baselines" />
<meta itemprop="path" content="stable" />
</div>

# utils.load_baselines

```python
utils.load_baselines(
    base_dir,
    verbose=False
)
```

Reads in the baseline experimental data from a specified base directory.

#### Args:

*   <b>`base_dir`</b>: string, base directory where to read data from.
*   <b>`verbose`</b>: bool, whether to print warning messages.

#### Returns:

A dict containing pandas DataFrames for all available agents and games.

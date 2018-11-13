<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="utils.load_statistics" />
<meta itemprop="path" content="stable" />
</div>

# utils.load_statistics

```python
utils.load_statistics(
    log_path,
    iteration_number=None,
    verbose=True
)
```

Reads in a statistics object from log_path.

#### Args:

*   <b>`log_path`</b>: string, provides the full path to the training/eval
    statistics.
*   <b>`iteration_number`</b>: The iteration number of the statistics object we
    want to read. If set to None, load the latest version.
*   <b>`verbose`</b>: Whether to output information about the load procedure.

#### Returns:

*   <b>`data`</b>: The requested statistics object.
*   <b>`iteration`</b>: The corresponding iteration number.

#### Raises:

*   <b>`Exception`</b>: if data is not present.

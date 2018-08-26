<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="utils.get_latest_iteration" />
<meta itemprop="path" content="stable" />
</div>

# utils.get_latest_iteration

```python
utils.get_latest_iteration(path)
```

Return the largest iteration number corresponding to the given path.

#### Args:

*   <b>`path`</b>: The base path (including directory and base name) to search.

#### Returns:

The latest iteration number.

#### Raises:

*   <b>`ValueError`</b>: if there is not available log data at the given path.

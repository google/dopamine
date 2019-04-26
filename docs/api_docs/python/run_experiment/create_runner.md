<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="run_experiment.create_runner" />
<meta itemprop="path" content="Stable" />
</div>

# run_experiment.create_runner

```python
run_experiment.create_runner(
    *args,
    **kwargs
)
```

Creates an experiment Runner.

#### Args:

*   <b>`base_dir`</b>: str, base directory for hosting all subdirectories.
*   <b>`schedule`</b>: string, which type of Runner to use.

#### Returns:

*   <b>`runner`</b>: A `Runner` like object.

#### Raises:

*   <b>`ValueError`</b>: When an unknown schedule is encountered.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="logger.Logger" />
<meta itemprop="path" content="stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__setitem__"/>
<meta itemprop="property" content="is_logging_enabled"/>
<meta itemprop="property" content="log_to_file"/>
</div>

# logger.Logger

## Class `Logger`

Class for maintaining a dictionary of data to log.

## Methods

<h3 id="__init__"><code>__init__</code></h3>

```python
__init__(logging_dir)
```

Initializes Logger.

#### Args:

*   <b>`logging_dir`</b>: str, Directory to which logs are written.

<h3 id="__setitem__"><code>__setitem__</code></h3>

```python
__setitem__(
    key,
    value
)
```

This method will set an entry at key with value in the dictionary.

It will effectively overwrite any previous data at the same key.

#### Args:

*   <b>`key`</b>: str, indicating key where to write the entry.
*   <b>`value`</b>: A python object to store.

<h3 id="is_logging_enabled"><code>is_logging_enabled</code></h3>

```python
is_logging_enabled()
```

Return if logging is enabled.

<h3 id="log_to_file"><code>log_to_file</code></h3>

```python
log_to_file(
    filename_prefix,
    iteration_number
)
```

Save the pickled dictionary to a file.

#### Args:

*   <b>`filename_prefix`</b>: str, name of the file to use (without iteration
    number).
*   <b>`iteration_number`</b>: int, the iteration number, appended to the end of
    filename_prefix.

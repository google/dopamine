<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="checkpointer.Checkpointer" />
<meta itemprop="path" content="stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="load_checkpoint"/>
<meta itemprop="property" content="save_checkpoint"/>
</div>

# checkpointer.Checkpointer

## Class `Checkpointer`

Class for managing checkpoints for Dopamine agents.

## Methods

<h3 id="__init__"><code>__init__</code></h3>

```python
__init__(
    base_directory,
    checkpoint_file_prefix='ckpt',
    checkpoint_frequency=1
)
```

Initializes Checkpointer.

#### Args:

*   <b>`base_directory`</b>: str, directory where all checkpoints are
    saved/loaded.
*   <b>`checkpoint_file_prefix`</b>: str, prefix to use for naming checkpoint
    files.
*   <b>`checkpoint_frequency`</b>: int, the frequency at which to checkpoint.

#### Raises:

*   <b>`ValueError`</b>: if base_directory is empty, or not creatable.

<h3 id="load_checkpoint"><code>load_checkpoint</code></h3>

```python
load_checkpoint(iteration_number)
```

Tries to reload a checkpoint at the selected iteration number.

#### Args:

*   <b>`iteration_number`</b>: The checkpoint iteration number to try to load.

#### Returns:

If the checkpoint files exist, two unpickled objects that were passed in as data
to save_checkpoint; returns None if the files do not exist.

<h3 id="save_checkpoint"><code>save_checkpoint</code></h3>

```python
save_checkpoint(
    iteration_number,
    data
)
```

Saves a new checkpoint at the current iteration_number.

#### Args:

*   <b>`iteration_number`</b>: int, the current iteration number for this
    checkpoint.
*   <b>`data`</b>: Any (picklable) python object containing the data to store in
    the checkpoint.

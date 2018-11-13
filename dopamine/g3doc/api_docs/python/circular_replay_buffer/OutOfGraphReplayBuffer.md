<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="circular_replay_buffer.OutOfGraphReplayBuffer" />
<meta itemprop="path" content="stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add"/>
<meta itemprop="property" content="cursor"/>
<meta itemprop="property" content="get_add_args_signature"/>
<meta itemprop="property" content="get_observation_stack"/>
<meta itemprop="property" content="get_range"/>
<meta itemprop="property" content="get_storage_signature"/>
<meta itemprop="property" content="get_terminal_stack"/>
<meta itemprop="property" content="get_transition_elements"/>
<meta itemprop="property" content="is_empty"/>
<meta itemprop="property" content="is_full"/>
<meta itemprop="property" content="is_valid_transition"/>
<meta itemprop="property" content="load"/>
<meta itemprop="property" content="sample_index_batch"/>
<meta itemprop="property" content="sample_transition_batch"/>
<meta itemprop="property" content="save"/>
</div>

# circular_replay_buffer.OutOfGraphReplayBuffer

## Class `OutOfGraphReplayBuffer`

A simple out-of-graph Replay Buffer.

Stores transitions, state, action, reward, next_state, terminal (and any extra
contents specified) in a circular buffer and provides a uniform transition
sampling function.

When the states consist of stacks of observations storing the states is
inefficient. This class writes observations and constructs the stacked states at
sample time.

#### Attributes:

*   <b>`add_count`</b>: int, counter of how many transitions have been added
    (including the blank ones at the beginning of an episode).

## Methods

<h3 id="__init__"><code>__init__</code></h3>

```python
__init__(
    observation_shape,
    stack_size,
    replay_capacity,
    batch_size,
    update_horizon=1,
    gamma=0.99,
    max_sample_attempts=MAX_SAMPLE_ATTEMPTS,
    extra_storage_types=None,
    observation_dtype=np.uint8
)
```

Initializes OutOfGraphReplayBuffer.

#### Args:

*   <b>`observation_shape`</b>: tuple or int. If int, the observation is assumed
    to be a 2D square.
*   <b>`stack_size`</b>: int, number of frames to use in state stack.
*   <b>`replay_capacity`</b>: int, number of transitions to keep in memory.
*   <b>`batch_size`</b>: int.
*   <b>`update_horizon`</b>: int, length of update ('n' in n-step update).
*   <b>`gamma`</b>: int, the discount factor.
*   <b>`max_sample_attempts`</b>: int, the maximum number of attempts allowed to
    get a sample.
*   <b>`extra_storage_types`</b>: list of ReplayElements defining the type of
    the extra contents that will be stored and returned by
    sample_transition_batch.
*   <b>`observation_dtype`</b>: np.dtype, type of the observations. Defaults to
    np.uint8 for Atari 2600.

#### Raises:

*   <b>`ValueError`</b>: If replay_capacity is too small to hold at least one
    transition.

<h3 id="add"><code>add</code></h3>

```python
add(
    observation,
    action,
    reward,
    terminal,
    *args
)
```

Adds a transition to the replay memory.

This function checks the types and handles the padding at the beginning of an
episode. Then it calls the _add function.

Since the next_observation in the transition will be the observation added next
there is no need to pass it.

If the replay memory is at capacity the oldest transition will be discarded.

#### Args:

*   <b>`observation`</b>: np.array with shape observation_shape.
*   <b>`action`</b>: int, the action in the transition.
*   <b>`reward`</b>: float, the reward received in the transition.
*   <b>`terminal`</b>: A uint8 acting as a boolean indicating whether the
    transition was terminal (1) or not (0).
*   <b>`*args`</b>: extra contents with shapes and dtypes according to
    extra_storage_types.

<h3 id="cursor"><code>cursor</code></h3>

```python
cursor()
```

Index to the location where the next transition will be written.

<h3 id="get_add_args_signature"><code>get_add_args_signature</code></h3>

```python
get_add_args_signature()
```

The signature of the add function.

Note - Derived classes may return a different signature.

#### Returns:

list of ReplayElements defining the type of the argument signature needed by the
add function.

<h3 id="get_observation_stack"><code>get_observation_stack</code></h3>

```python
get_observation_stack(index)
```

<h3 id="get_range"><code>get_range</code></h3>

```python
get_range(
    array,
    start_index,
    end_index
)
```

Returns the range of array at the index handling wraparound if necessary.

#### Args:

*   <b>`array`</b>: np.array, the array to get the stack from.
*   <b>`start_index`</b>: int, index to the start of the range to be returned.
    Range will wraparound if start_index is smaller than 0.
*   <b>`end_index`</b>: int, exclusive end index. Range will wraparound if
    end_index exceeds replay_capacity.

#### Returns:

np.array, with shape [end_index - start_index, array.shape[1:]].

<h3 id="get_storage_signature"><code>get_storage_signature</code></h3>

```python
get_storage_signature()
```

Returns a default list of elements to be stored in this replay memory.

Note - Derived classes may return a different signature.

#### Returns:

list of ReplayElements defining the type of the contents stored.

<h3 id="get_terminal_stack"><code>get_terminal_stack</code></h3>

```python
get_terminal_stack(index)
```

<h3 id="get_transition_elements"><code>get_transition_elements</code></h3>

```python
get_transition_elements(batch_size=None)
```

Returns a 'type signature' for sample_transition_batch.

#### Args:

*   <b>`batch_size`</b>: int, number of transitions returned. If None, the
    default batch_size will be used.

#### Returns:

*   <b>`signature`</b>: A namedtuple describing the method's return type
    signature.

<h3 id="is_empty"><code>is_empty</code></h3>

```python
is_empty()
```

Is the Replay Buffer empty?

<h3 id="is_full"><code>is_full</code></h3>

```python
is_full()
```

Is the Replay Buffer full?

<h3 id="is_valid_transition"><code>is_valid_transition</code></h3>

```python
is_valid_transition(index)
```

Checks if the index contains a valid transition.

Checks for collisions with the end of episodes and the current position of the
cursor.

#### Args:

*   <b>`index`</b>: int, the index to the state in the transition.

#### Returns:

Is the index valid: Boolean.

<h3 id="load"><code>load</code></h3>

```python
load(
    checkpoint_dir,
    suffix
)
```

Restores the object from bundle_dictionary and numpy checkpoints.

#### Args:

*   <b>`checkpoint_dir`</b>: str, the directory where to read the numpy
    checkpointed files from.
*   <b>`suffix`</b>: str, the suffix to use in numpy checkpoint files.

#### Raises:

*   <b>`NotFoundError`</b>: If not all expected files are found in directory.

<h3 id="sample_index_batch"><code>sample_index_batch</code></h3>

```python
sample_index_batch(batch_size)
```

Returns a batch of valid indices sampled uniformly.

#### Args:

*   <b>`batch_size`</b>: int, number of indices returned.

#### Returns:

list of ints, a batch of valid indices sampled uniformly.

#### Raises:

*   <b>`RuntimeError`</b>: If the batch was not constructed after maximum number
    of tries.

<h3 id="sample_transition_batch"><code>sample_transition_batch</code></h3>

```python
sample_transition_batch(
    batch_size=None,
    indices=None
)
```

Returns a batch of transitions (including any extra contents).

If get_transition_elements has been overridden and defines elements not stored
in self._store, an empty array will be returned and it will be left to the child
class to fill it. For example, for the child class
OutOfGraphPrioritizedReplayBuffer, the contents of the sampling_probabilities
are stored separately in a sum tree.

When the transition is terminal next_state_batch has undefined contents.

NOTE: This transition contains the indices of the sampled elements. These are
only valid during the call to sample_transition_batch, i.e. they may be used by
subclasses of this replay buffer but may point to different data as soon as
sampling is done.

#### Args:

*   <b>`batch_size`</b>: int, number of transitions returned. If None, the
    default batch_size will be used.
*   <b>`indices`</b>: None or list of ints, the indices of every transition in
    the batch. If None, sample the indices uniformly.

#### Returns:

*   <b>`transition_batch`</b>: tuple of np.arrays with the shape and type as in
    get_transition_elements().

#### Raises:

*   <b>`ValueError`</b>: If an element to be sampled is missing from the replay
    buffer.

<h3 id="save"><code>save</code></h3>

```python
save(
    checkpoint_dir,
    iteration_number
)
```

Save the OutOfGraphReplayBuffer attributes into a file.

This method will save all the replay buffer's state in a single file.

#### Args:

*   <b>`checkpoint_dir`</b>: str, the directory where numpy checkpoint files
    should be saved.
*   <b>`iteration_number`</b>: int, iteration_number to use as a suffix in
    naming numpy checkpoint files.

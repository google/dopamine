<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="prioritized_replay_buffer.WrappedPrioritizedReplayBuffer" />
<meta itemprop="path" content="stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add"/>
<meta itemprop="property" content="create_sampling_ops"/>
<meta itemprop="property" content="load"/>
<meta itemprop="property" content="save"/>
<meta itemprop="property" content="tf_get_priority"/>
<meta itemprop="property" content="tf_set_priority"/>
<meta itemprop="property" content="unpack_transition"/>
</div>

# prioritized_replay_buffer.WrappedPrioritizedReplayBuffer

## Class `WrappedPrioritizedReplayBuffer`

Inherits From:
[`WrappedReplayBuffer`](../circular_replay_buffer/WrappedReplayBuffer.md)

Wrapper of OutOfGraphPrioritizedReplayBuffer with in-graph sampling.

Usage:

*   To add a transition: Call the add function.

*   To sample a batch: Query any of the tensors in the transition dictionary.
    Every sess.run that requires any of these tensors will sample a new
    transition.

## Methods

<h3 id="__init__"><code>__init__</code></h3>

```python
__init__(
    *args,
    **kwargs
)
```

Initializes WrappedPrioritizedReplayBuffer.

#### Args:

*   <b>`observation_shape`</b>: tuple or int. If int, the observation is assumed
    to be a 2D square with sides equal to observation_shape.
*   <b>`stack_size`</b>: int, number of frames to use in state stack.
*   <b>`use_staging`</b>: bool, when True it would use a staging area to
    prefetch the next sampling batch.
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

*   <b>`ValueError`</b>: If update_horizon is not positive.
*   <b>`ValueError`</b>: If discount factor is not in [0, 1].

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

<h3 id="create_sampling_ops"><code>create_sampling_ops</code></h3>

```python
create_sampling_ops(use_staging)
```

Creates the ops necessary to sample from the replay buffer.

Creates the transition dictionary containing the sampling tensors.

#### Args:

*   <b>`use_staging`</b>: bool, when True it would use a staging area to
    prefetch the next sampling batch.

<h3 id="load"><code>load</code></h3>

```python
load(
    checkpoint_dir,
    suffix
)
```

Loads the replay buffer's state from a saved file.

#### Args:

*   <b>`checkpoint_dir`</b>: str, the directory where to read the numpy
    checkpointed files from.
*   <b>`suffix`</b>: str, the suffix to use in numpy checkpoint files.

<h3 id="save"><code>save</code></h3>

```python
save(
    checkpoint_dir,
    iteration_number
)
```

Save the underlying replay buffer's contents in a file.

#### Args:

*   <b>`checkpoint_dir`</b>: str, the directory where to read the numpy
    checkpointed files from.
*   <b>`iteration_number`</b>: int, the iteration_number to use as a suffix in
    naming numpy checkpoint files.

<h3 id="tf_get_priority"><code>tf_get_priority</code></h3>

```python
tf_get_priority(indices)
```

Gets the priorities for the given indices.

#### Args:

*   <b>`indices`</b>: tf.Tensor with dtype int32 and shape [n].

#### Returns:

*   <b>`priorities`</b>: tf.Tensor with dtype float and shape [n], the
    priorities at the indices.

<h3 id="tf_set_priority"><code>tf_set_priority</code></h3>

```python
tf_set_priority(
    indices,
    priorities
)
```

Sets the priorities for the given indices.

#### Args:

*   <b>`indices`</b>: tf.Tensor with dtype int32 and shape [n].
*   <b>`priorities`</b>: tf.Tensor with dtype float and shape [n].

#### Returns:

A tf op setting the priorities for prioritized sampling.

<h3 id="unpack_transition"><code>unpack_transition</code></h3>

```python
unpack_transition(
    transition_tensors,
    transition_type
)
```

Unpacks the given transition into member variables.

#### Args:

*   <b>`transition_tensors`</b>: tuple of tf.Tensors.
*   <b>`transition_type`</b>: tuple of ReplayElements matching
    transition_tensors.

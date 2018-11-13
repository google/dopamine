<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="rainbow_agent.project_distribution" />
<meta itemprop="path" content="stable" />
</div>

# rainbow_agent.project_distribution

```python
rainbow_agent.project_distribution(
    supports,
    weights,
    target_support,
    validate_args=False
)
```

Projects a batch of (support, weights) onto target_support.

Based on equation (7) in (Bellemare et al., 2017):
https://arxiv.org/abs/1707.06887 In the rest of the comments we will refer to
this equation simply as Eq7.

This code is not easy to digest, so we will use a running example to clarify
what is going on, with the following sample inputs:

*   supports = [[0, 2, 4, 6, 8], [1, 3, 4, 5, 6]]
*   weights = [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.2, 0.5, 0.1, 0.1]]
*   target_support = [4, 5, 6, 7, 8]

In the code below, comments preceded with 'Ex:' will be referencing the above
values.

#### Args:

*   <b>`supports`</b>: Tensor of shape (batch_size, num_dims) defining supports
    for the distribution.
*   <b>`weights`</b>: Tensor of shape (batch_size, num_dims) defining weights on
    the original support points. Although for the CategoricalDQN agent these
    weights are probabilities, it is not required that they are.
*   <b>`target_support`</b>: Tensor of shape (num_dims) defining support of the
    projected distribution. The values must be monotonically increasing. Vmin
    and Vmax will be inferred from the first and last elements of this tensor,
    respectively. The values in this tensor must be equally spaced.
*   <b>`validate_args`</b>: Whether we will verify the contents of the
    target_support parameter.

#### Returns:

A Tensor of shape (batch_size, num_dims) with the projection of a batch of
(support, weights) onto target_support.

#### Raises:

*   <b>`ValueError`</b>: If target_support has no dimensions, or if shapes of
    supports, weights, and target_support are incompatible.

description: MessagePack serialization hooks.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.jax.serialization" />
<meta itemprop="path" content="Stable" />
</div>

# Module: dopamine.jax.serialization

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/jax/serialization.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



MessagePack serialization hooks.


This is necesarry as Orbax doesn't support serializing Numpy structured arrays.
TensorStore (the underlying backend for Orbax) does support structured arrays.
If Orbax ever adds support for structured arrays, we can remove this altogether
and use Orbax to serialize things like the replay buffer without requiring
a custom checkpoint handler (jax/checkpointers.py).

## Classes

[`class LongIntegerEncoding`](../../dopamine/jax/serialization/LongIntegerEncoding.md): Encoding for ints longer than 32-bits which MessagePack doesn't support.

[`class NumpyEncoding`](../../dopamine/jax/serialization/NumpyEncoding.md): Numpy encoding dictionary.

## Functions

[`decode(...)`](../../dopamine/jax/serialization/decode.md): Decode encoded object types.

[`encode(...)`](../../dopamine/jax/serialization/encode.md): Encode object. Encoders will register with `@encode.register`.


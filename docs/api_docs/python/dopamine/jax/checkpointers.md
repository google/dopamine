description: Checkpointer using Orbax and MessagePack.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.jax.checkpointers" />
<meta itemprop="path" content="Stable" />
</div>

# Module: dopamine.jax.checkpointers

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/jax/checkpointers.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Checkpointer using Orbax and MessagePack.



#### Usage:

1) Implement `to_state_dict` and `from_state_dict` on your class.
2) Create an `orbax.CheckpointManager` and use the `CheckpointHandler`.

To save: call `save` on your checkpoint manager passing the class instance.
To restore: call `restore` on your checkpoint manager passing the class
            instance.



#### Example:


```py
from typing import Dict, Any

import numpy.typing as npt
from orbax import checkpoint as orbax
from dopamine.jax import checkpoint

class MyClass:
  def __init__(self, data: npt.NDArray):
    self._data = data

  def to_state_dict(self) -> Dict[str, Any]:
    return {'data': self._data}

  def from_state_dict(self, state_dict: Dict[str, Any]) -> None:
    self._data = state_dict['data']

checkpoint_manager = orbax.CheckpointManager(
    my_workdir,
    checkpointers=checkpoint.Checkpointer(),
)

instance = MyClass(np.array([1, 2, 3]))
checkpoint_manager.save(1, instance)
# Creates a copy of instance
restored_instance = checkpoint_manager.restore(1, instance)

# If you don't want to create a copy:
instance.from_state_dict(checkpoint_manager.restore(1))
```


## Classes

[`class CheckpointHandler`](../../dopamine/jax/checkpointers/CheckpointHandler.md): Checkpointable protocol checkpoint handler.

[`class Checkpointable`](../../dopamine/jax/checkpointers/Checkpointable.md): Checkpointable protocol. Must implement to_state_dict, from_state_dict.

## Functions

[`Checkpointer(...)`](../../dopamine/jax/checkpointers/Checkpointer.md): partial(func, *args, **keywords) - new function with partial application of the given arguments and keywords.


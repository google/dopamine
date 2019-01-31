<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="atari_lib.create_atari_environment" />
<meta itemprop="path" content="Stable" />
</div>

# atari_lib.create_atari_environment

```python
atari_lib.create_atari_environment(
    *args,
    **kwargs
)
```

Wraps an Atari 2600 Gym environment with some basic preprocessing.

This preprocessing matches the guidelines proposed in Machado et al. (2017),
"Revisiting the Arcade Learning Environment: Evaluation Protocols and Open
Problems for General Agents".

The created environment is the Gym wrapper around the Arcade Learning
Environment.

The main choice available to the user is whether to use sticky actions or not.
Sticky actions, as prescribed by Machado et al., cause actions to persist with
some probability (0.25) when a new command is sent to the ALE. This can be
viewed as introducing a mild form of stochasticity in the environment. We use
them by default.

#### Args:

*   <b>`game_name`</b>: str, the name of the Atari 2600 domain.
*   <b>`sticky_actions`</b>: bool, whether to use sticky_actions as per Machado
    et al.

#### Returns:

An Atari 2600 environment with some standard preprocessing.

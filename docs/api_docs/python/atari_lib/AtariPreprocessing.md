<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="atari_lib.AtariPreprocessing" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="action_space"/>
<meta itemprop="property" content="metadata"/>
<meta itemprop="property" content="observation_space"/>
<meta itemprop="property" content="reward_range"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="render"/>
<meta itemprop="property" content="reset"/>
<meta itemprop="property" content="step"/>
</div>

# atari_lib.AtariPreprocessing

## Class `AtariPreprocessing`

A class implementing image preprocessing for Atari 2600 agents.

Specifically, this provides the following subset from the JAIR paper (Bellemare
et al., 2013) and Nature DQN paper (Mnih et al., 2015):

*   Frame skipping (defaults to 4).
*   Terminal signal when a life is lost (off by default).
*   Grayscale and max-pooling of the last two frames.
*   Downsample the screen to a square image (defaults to 84x84).

More generally, this class follows the preprocessing guidelines set down in
Machado et al. (2018), "Revisiting the Arcade Learning Environment: Evaluation
Protocols and Open Problems for General Agents".

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(
    *args,
    **kwargs
)
```

Constructor for an Atari 2600 preprocessor.

#### Args:

*   <b>`environment`</b>: Gym environment whose observations are preprocessed.
*   <b>`frame_skip`</b>: int, the frequency at which the agent experiences the
    game.
*   <b>`terminal_on_life_loss`</b>: bool, If True, the step() method returns
    is_terminal=True whenever a life is lost. See Mnih et al. 2015.
*   <b>`screen_size`</b>: int, size of a resized Atari 2600 frame.

#### Raises:

*   <b>`ValueError`</b>: if frame_skip or screen_size are not strictly positive.

## Properties

<h3 id="action_space"><code>action_space</code></h3>

<h3 id="metadata"><code>metadata</code></h3>

<h3 id="observation_space"><code>observation_space</code></h3>

<h3 id="reward_range"><code>reward_range</code></h3>

## Methods

<h3 id="render"><code>render</code></h3>

```python
render(mode)
```

Renders the current screen, before preprocessing.

This calls the Gym API's render() method.

#### Args:

*   <b>`mode`</b>: Mode argument for the environment's render() method. Valid
    values (str) are: 'rgb_array': returns the raw ALE image. 'human': renders
    to display via the Gym renderer.

#### Returns:

if mode='rgb_array': numpy array, the most recent screen. if mode='human': bool,
whether the rendering was successful.

<h3 id="reset"><code>reset</code></h3>

```python
reset()
```

Resets the environment.

#### Returns:

*   <b>`observation`</b>: numpy array, the initial observation emitted by the
    environment.

<h3 id="step"><code>step</code></h3>

```python
step(action)
```

Applies the given action in the environment.

Remarks:

*   If a terminal state (from life loss or episode end) is reached, this may
    execute fewer than self.frame_skip steps in the environment.
*   Furthermore, in this case the returned observation may not contain valid
    image data and should be ignored.

#### Args:

*   <b>`action`</b>: The action to be executed.

#### Returns:

*   <b>`observation`</b>: numpy array, the observation following the action.
*   <b>`reward`</b>: float, the reward following the action.
*   <b>`is_terminal`</b>: bool, whether the environment has reached a terminal
    state. This is true when a life is lost and terminal_on_life_loss, or when
    the episode is over.
*   <b>`info`</b>: Gym API's info data structure.

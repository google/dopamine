description: Gym-specific (non-Atari) utilities.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.discrete_domains.gym_lib" />
<meta itemprop="path" content="Stable" />
</div>

# Module: dopamine.discrete_domains.gym_lib

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/discrete_domains/gym_lib.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Gym-specific (non-Atari) utilities.

Some network specifications specific to certain Gym environments are provided
here.

Includes a wrapper class around Gym environments. This class makes general Gym
environments conformant with the API Dopamine is expecting.

## Classes

[`class GymPreprocessing`](../../dopamine/discrete_domains/gym_lib/GymPreprocessing.md):
A Wrapper class around Gym environments.

## Functions

[`create_gym_environment(...)`](../../dopamine/discrete_domains/gym_lib/create_gym_environment.md):
Wraps a Gym environment with some basic preprocessing.

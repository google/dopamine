description: A class implementing image preprocessing for Atari 2600 agents.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.discrete_domains.atari_lib.AtariPreprocessing" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.discrete_domains.atari_lib.AtariPreprocessing

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/discrete_domains/atari_lib.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

A class implementing image preprocessing for Atari 2600 agents.

<!-- Placeholder for "Used in" -->

Specifically, this provides the following subset from the JAIR paper (Bellemare
et al., 2013) and Nature DQN paper (Mnih et al., 2015):

*   Frame skipping (defaults to 4).
*   Terminal signal when a life is lost (off by default).
*   Grayscale and max-pooling of the last two frames.
*   Downsample the screen to a square image (defaults to 84x84).

More generally, this class follows the preprocessing guidelines set down in
Machado et al. (2018), "Revisiting the Arcade Learning Environment: Evaluation
Protocols and Open Problems for General Agents".

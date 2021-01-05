description: Wraps an Atari 2600 Gym environment with some basic preprocessing.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.discrete_domains.atari_lib.create_atari_environment" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.discrete_domains.atari_lib.create_atari_environment

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/discrete_domains/atari_lib.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Wraps an Atari 2600 Gym environment with some basic preprocessing.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dopamine.discrete_domains.atari_lib.create_atari_environment(
    game_name=None, sticky_actions=True
)
</code></pre>

<!-- Placeholder for "Used in" -->

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

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`game_name`
</td>
<td>
str, the name of the Atari 2600 domain.
</td>
</tr><tr>
<td>
`sticky_actions`
</td>
<td>
bool, whether to use sticky_actions as per Machado et al.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An Atari 2600 environment with some standard preprocessing.
</td>
</tr>

</table>

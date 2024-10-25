description: Library used by example_viz.py to generate visualizations.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.utils.example_viz_lib" />
<meta itemprop="path" content="Stable" />
</div>

# Module: dopamine.utils.example_viz_lib

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/utils/example_viz_lib.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Library used by example_viz.py to generate visualizations.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`dopamine.utils.example_viz.example_viz_lib`</p>
</p>
</section>


This file illustrates the following:
  - How to subclass an existing agent to add visualization functionality.
    - For DQN we visualize the cumulative rewards and the Q-values for each
      action (MyDQNAgent).
    - For Rainbow we visualize the cumulative rewards and the Q-value
      distributions for each action (MyRainbowAgent).
  - How to subclass Runner to run in eval mode, lay out the different subplots,
    generate the visualizations, and compile them into a video (MyRunner).
  - The function `run()` is the main entrypoint for running everything.

## Functions

[`create_runner(...)`](../../dopamine/utils/example_viz_lib/create_runner.md)


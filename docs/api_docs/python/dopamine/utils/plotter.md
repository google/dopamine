description: Base class for plotters.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.utils.plotter" />
<meta itemprop="path" content="Stable" />
</div>

# Module: dopamine.utils.plotter

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/utils/plotter.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Base class for plotters.


This class provides the core functionality for Plotter objects. Specifically, it
initializes `self.parameters` with the values passed through the constructor or
with the provided defaults (specified in each child class), and specifies the
abstract `draw()` method, which child classes will need to implement.

This class also provides a helper function `_setup_plot` for Plotters based on
matplotlib.

## Classes

[`class Plotter`](../../dopamine/utils/plotter/Plotter.md): Abstract base class for plotters.


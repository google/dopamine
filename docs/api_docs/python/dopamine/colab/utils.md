description: This provides utilities for dealing with Dopamine data.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.colab.utils" />
<meta itemprop="path" content="Stable" />
</div>

# Module: dopamine.colab.utils

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/colab/utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

This provides utilities for dealing with Dopamine data.

See: dopamine/common/logger.py .

## Functions

[`get_latest_file(...)`](../../dopamine/colab/utils/get_latest_file.md): Return
the file named 'path_[0-9]*' with the largest such number.

[`get_latest_iteration(...)`](../../dopamine/colab/utils/get_latest_iteration.md):
Return the largest iteration number corresponding to the given path.

[`load_baselines(...)`](../../dopamine/colab/utils/load_baselines.md): Reads in
the baseline experimental data from a specified base directory.

[`load_statistics(...)`](../../dopamine/colab/utils/load_statistics.md): Reads
in a statistics object from log_path.

[`read_experiment(...)`](../../dopamine/colab/utils/read_experiment.md): Reads
in a set of experimental results from log_path.

[`summarize_data(...)`](../../dopamine/colab/utils/summarize_data.md): Processes
log data into a per-iteration summary.

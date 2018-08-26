<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="utils" />
<meta itemprop="path" content="stable" />
</div>

# Module: utils

This provides utilities for dealing with Dopamine data.

See: dopamine/common/logger.py .

## Functions

[`get_latest_file(...)`](./utils/get_latest_file.md): Return the file named
'path_[0-9]*' with the largest such number.

[`get_latest_iteration(...)`](./utils/get_latest_iteration.md): Return the
largest iteration number corresponding to the given path.

[`load_baselines(...)`](./utils/load_baselines.md): Reads in the baseline
experimental data from a specified base directory.

[`load_statistics(...)`](./utils/load_statistics.md): Reads in a statistics
object from log_path.

[`read_experiment(...)`](./utils/read_experiment.md): Reads in a set of
experimental results from log_path.

[`summarize_data(...)`](./utils/summarize_data.md): Processes log data into a
per-iteration summary.

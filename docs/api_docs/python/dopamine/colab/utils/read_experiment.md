description: Reads in a set of experimental results from log_path.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.colab.utils.read_experiment" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.colab.utils.read_experiment

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/colab/utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Reads in a set of experimental results from log_path.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dopamine.colab.utils.read_experiment(
    log_path, parameter_set=None, job_descriptor='', iteration_number=None,
    summary_keys=('train_episode_returns', 'eval_episode_returns'), verbose=False
)
</code></pre>

<!-- Placeholder for "Used in" -->

The provided parameter_set is an ordered_dict which 1) defines the parameters of
this experiment, 2) defines the order in which they occur in the job descriptor.

The method reads all experiments of the form

${log_path}/${job_descriptor}.format(params)/logs,

where params is constructed from the cross product of the elements in the
parameter_set.

#### For example:

parameter_set = collections.OrderedDict([ ('game', ['Asterix', 'Pong']),
('epsilon', ['0', '0.1']) ]) read_experiment('/tmp/logs', parameter_set,
job_descriptor='{}_{}') Will try to read logs from: - /tmp/logs/Asterix_0/logs -
/tmp/logs/Asterix_0.1/logs - /tmp/logs/Pong_0/logs - /tmp/logs/Pong_0.1/logs

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`log_path`
</td>
<td>
string, base path specifying where results live.
</td>
</tr><tr>
<td>
`parameter_set`
</td>
<td>
An ordered_dict mapping parameter names to allowable values.
</td>
</tr><tr>
<td>
`job_descriptor`
</td>
<td>
A job descriptor string which is used to construct the full
path for each trial within an experiment.
</td>
</tr><tr>
<td>
`iteration_number`
</td>
<td>
Int, if not None determines the iteration number at which
we read in results.
</td>
</tr><tr>
<td>
`summary_keys`
</td>
<td>
Iterable of strings, iteration statistics to summarize.
</td>
</tr><tr>
<td>
`verbose`
</td>
<td>
If True, print out additional information.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Pandas dataframe containing experimental results.
</td>
</tr>

</table>

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="utils.read_experiment" />
<meta itemprop="path" content="stable" />
</div>

# utils.read_experiment

```python
utils.read_experiment(
    log_path,
    parameter_set=None,
    job_descriptor='',
    iteration_number=None,
    summary_keys=('train_episode_returns', 'eval_episode_returns'),
    verbose=False
)
```

Reads in a set of experimental results from log_path.

The provided parameter_set is an ordered_dict which 1) defines the parameters of
this experiment, 2) defines the order in which they occur in the job descriptor.

The method reads all experiments of the form

${log_path}/${job_descriptor}.format(params)/logs,

where params is constructed from the cross product of the elements in the
parameter_set.

For example: parameter_set = collections.OrderedDict([ ('game', ['Asterix',
'Pong']), ('epsilon', ['0', '0.1']) ]) read_experiment('/tmp/logs',
parameter_set, job_descriptor='{}_{}') Will try to read logs from: -
/tmp/logs/Asterix_0/logs - /tmp/logs/Asterix_0.1/logs - /tmp/logs/Pong_0/logs -
/tmp/logs/Pong_0.1/logs

#### Args:

*   <b>`log_path`</b>: string, base path specifying where results live.
*   <b>`parameter_set`</b>: An ordered_dict mapping parameter names to allowable
    values.
*   <b>`job_descriptor`</b>: A job descriptor string which is used to construct
    the full path for each trial within an experiment.
*   <b>`iteration_number`</b>: Int, if not None determines the iteration number
    at which we read in results.
*   <b>`summary_keys`</b>: Iterable of strings, iteration statistics to
    summarize.
*   <b>`verbose`</b>: If True, print out additional information.

#### Returns:

A Pandas dataframe containing experimental results.

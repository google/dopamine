description: A checkpointing mechanism for Dopamine agents.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.discrete_domains.checkpointer" />
<meta itemprop="path" content="Stable" />
</div>

# Module: dopamine.discrete_domains.checkpointer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/discrete_domains/checkpointer.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

A checkpointing mechanism for Dopamine agents.

This Checkpointer expects a base directory where checkpoints for different
iterations are stored. Specifically, Checkpointer.save_checkpoint() takes in as
input a dictionary 'data' to be pickled to disk. At each iteration, we write a
file called 'cpkt.#', where # is the iteration number. The Checkpointer also
cleans up old files, maintaining up to the CHECKPOINT_DURATION most recent
iterations.

The Checkpointer writes a sentinel file to indicate that checkpointing was
globally successful. This means that all other checkpointing activities (saving
the Tensorflow graph, the replay buffer) should be performed *prior* to calling
Checkpointer.save_checkpoint(). This allows the Checkpointer to detect
incomplete checkpoints.

#### Example

After running 10 iterations (numbered 0...9) with base_directory='/checkpoint',
the following files will exist: `/checkpoint/cpkt.6 /checkpoint/cpkt.7
/checkpoint/cpkt.8 /checkpoint/cpkt.9 /checkpoint/sentinel_checkpoint_complete.6
/checkpoint/sentinel_checkpoint_complete.7
/checkpoint/sentinel_checkpoint_complete.8
/checkpoint/sentinel_checkpoint_complete.9`

## Classes

[`class Checkpointer`](../../dopamine/discrete_domains/checkpointer/Checkpointer.md):
Class for managing checkpoints for Dopamine agents.

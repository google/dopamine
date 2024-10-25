description: Sample file to generate visualizations.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.utils.example_viz" />
<meta itemprop="path" content="Stable" />
</div>

# Module: dopamine.utils.example_viz

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/utils/example_viz.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Sample file to generate visualizations.


To run, point FLAGS.restore_checkpoint to the TensorFlow checkpoint of a
trained agent. As an example, you can download to `/tmp/checkpoints` the files
linked below:

  * https://storage.cloud.google.com/download-dopamine-rl/colab/samples/rainbow/SpaceInvaders_v4/checkpoints/tf_ckpt-199.data-00000-of-00001
  * https://storage.cloud.google.com/download-dopamine-rl/colab/samples/rainbow/SpaceInvaders_v4/checkpoints/tf_ckpt-199.index
  * https://storage.cloud.google.com/download-dopamine-rl/colab/samples/rainbow/SpaceInvaders_v4/checkpoints/tf_ckpt-199.meta


You can then run the binary with:

```
python example_viz.py \
        --agent='rainbow' \
        --game='SpaceInvaders' \
        --num_steps=1000 \
        --root_dir='/tmp/dopamine' \
        --restore_checkpoint=/tmp/checkpoints/colab_samples_rainbow_SpaceInvaders_v4_checkpoints_tf_ckpt-199
```

## Modules

[`example_viz_lib`](../../dopamine/utils/example_viz_lib.md) module: Library used by example_viz.py to generate visualizations.


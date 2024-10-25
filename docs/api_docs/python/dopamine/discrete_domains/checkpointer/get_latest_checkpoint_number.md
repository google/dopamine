description: Returns the version number of the latest completed checkpoint.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.discrete_domains.checkpointer.get_latest_checkpoint_number" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.discrete_domains.checkpointer.get_latest_checkpoint_number

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/discrete_domains/checkpointer.py#L57-L90">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns the version number of the latest completed checkpoint.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dopamine.discrete_domains.checkpointer.get_latest_checkpoint_number(
    base_directory,
    override_number=None,
    sentinel_file_identifier=&#x27;checkpoint&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`base_directory`<a id="base_directory"></a>
</td>
<td>
str, directory in which to look for checkpoint files.
</td>
</tr><tr>
<td>
`override_number`<a id="override_number"></a>
</td>
<td>
None or int, allows the user to manually override the
checkpoint number via a gin-binding.
</td>
</tr><tr>
<td>
`sentinel_file_identifier`<a id="sentinel_file_identifier"></a>
</td>
<td>
str, prefix used by checkpointer for naming
sentinel files.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
int, the iteration number of the latest checkpoint, or -1 if none was found.
</td>
</tr>

</table>


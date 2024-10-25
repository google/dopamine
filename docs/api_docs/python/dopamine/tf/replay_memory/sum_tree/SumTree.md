description: A sum tree data structure for storing replay priorities.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.tf.replay_memory.sum_tree.SumTree" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.tf.replay_memory.sum_tree.SumTree

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/tf/replay_memory/sum_tree.py#L29-L209">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A sum tree data structure for storing replay priorities.

<!-- Placeholder for "Used in" -->

A sum tree is a complete binary tree whose leaves contain values called
priorities. Internal nodes maintain the sum of the priorities of all leaf
nodes in their subtree.

For capacity = 4, the tree may look like this:

             +---+
             |2.5|
             +-+-+
               |
       +-------+--------+
       |                |
     +-+-+            +-+-+
     |1.5|            |1.0|
     +-+-+            +-+-+
       |                |
  +----+----+      +----+----+
  |         |      |         |
+-+-+     +-+-+  +-+-+     +-+-+
|0.5|     |1.0|  |0.5|     |0.5|
+---+     +---+  +---+     +---+

This is stored in a list of numpy arrays:
self.nodes = [ [2.5], [1.5, 1], [0.5, 1, 0.5, 0.5] ]

For conciseness, we allocate arrays as powers of two, and pad the excess
elements with zero values.

This is similar to the usual array-based representation of a complete binary
tree, but is a little more user-friendly.


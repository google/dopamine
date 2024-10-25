description: The fully connected network used to compute the agent's Q-values.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.discrete_domains.gym_lib.BasicDiscreteDomainNetwork" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.discrete_domains.gym_lib.BasicDiscreteDomainNetwork

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/discrete_domains/gym_lib.py#L107-L167">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The fully connected network used to compute the agent's Q-values.

<!-- Placeholder for "Used in" -->

This sub network used within various other models. Since it is an inner
block, we define it as a layer. These sub networks normalize their inputs to
lie in range [-1, 1], using min_/max_vals. It supports both DQN- and
Rainbow- style networks.
Attributes:
  min_vals: float, minimum attainable values (must be same shape as `state`).
  max_vals: float, maximum attainable values (must be same shape as `state`).
  num_actions: int, number of actions.
  num_atoms: int or None, if None will construct a DQN-style network,
    otherwise will construct a Rainbow-style network.
  name: str, used to create scope for network parameters.
  activation_fn: function, passed to the layer constructors.


description: Fourier Basis linear function approximation.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.discrete_domains.gym_lib.FourierBasis" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.discrete_domains.gym_lib.FourierBasis

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/discrete_domains/gym_lib.py#L194-L229">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Fourier Basis linear function approximation.

<!-- Placeholder for "Used in" -->

Requires the ranges for each dimension, and is thus able to use only sine or
cosine (and uses cosine). So, this has half the coefficients that a full
Fourier approximation would use.

Many thanks to Will Dabney (wdabney@) for this implementation.

#### From the paper:


G.D. Konidaris, S. Osentoski and P.S. Thomas. (2011)
Value Function Approximation in Reinforcement Learning using the Fourier Basis


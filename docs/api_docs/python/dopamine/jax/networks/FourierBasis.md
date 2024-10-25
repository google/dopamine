description: Fourier Basis linear function approximation.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="dopamine.jax.networks.FourierBasis" />
<meta itemprop="path" content="Stable" />
</div>

# dopamine.jax.networks.FourierBasis

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/dopamine/tree/master/dopamine/jax/networks.py#L221-L266">
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

Adapted from Will Dabney's (wdabney@) TF implementation for JAX.

#### From the paper:


G.D. Konidaris, S. Osentoski and P.S. Thomas. (2011)
Value Function Approximation in Reinforcement Learning using the Fourier Basis


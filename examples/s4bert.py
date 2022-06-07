"""
This example creates a simple, lightweight sequence transduction model backbone
by stacking S4D blocks in a manner not dissimilar from an encoder-only
transformer such as BERT-- then runs a forward pass with the resulting model.

This implementation omits the embedding table and tokenization typical of
language models for simplicity and generality. It is also comparatively much
simpler than the usual transformer equivalent because it incorporates sequential
inductive bias, eliminating the need for complex positional encoding schemes.
Its parameter count weighs in at about half the size of a BERT model with the
same number of attention heads (state channels here), hidden dimensions, and
layers, or residual blocks.

TODO(kavorite): implement multi-scale hierarchical architecture 
    https://arxiv.org/abs/2110.13711
    https://arxiv.org/abs/2202.09729
"""

import haiku as hk
import jax.numpy as jnp
from jax.random import PRNGKey
from s4d import S4DEncoder


@hk.without_apply_rng
@hk.transform
def model(u, state=None, **kwargs):
    A, L, H = 4, 12, 256  # electra-small-decoder configuration
    d = u.shape[-1]
    u = hk.Linear(H)(u)
    layers = [S4DEncoder(H, A) for _ in range(L)]
    cnn = hk.Sequential(layers)
    rnn = hk.DeepRNN(layers)

    if state is None:
        v = cnn(u)
    else:
        v, state = hk.dynamic_unroll(rnn, u, state, **kwargs)
    y = hk.Linear(d)(v)
    return y if state is None else (y, state)


u = jnp.zeros((1, 512, 256))  # (B N D)
params = model.init(PRNGKey(42), u)
print(f"{hk.data_structures.tree_size(params) / 1e6:.3g}M parameters")
v = model.apply(params, u)  # apply convolutional parametrization
pass

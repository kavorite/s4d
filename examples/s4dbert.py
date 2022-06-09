"""
This example creates a simple, lightweight sequence transduction model backbone
by stacking S4D blocks in a manner not dissimilar from an encoder-only
transformer such as BERT-- then runs a forward pass with the resulting model.

This implementation omits the embedding table and tokenization typical of
language models for simplicity and generality. It is also comparatively much
more concise than its encoder-only attention-based equivalent because its
sequential inductive bias eliminates the need for complex positional encoding
schemes. Its parameter count weighs in at about 40% larger than a BERT model
with the same number of attention heads (state channels here), hidden
dimensions, and residual blocks.

TODO(kavorite): implement multi-scale hierarchical architecture 
    https://arxiv.org/abs/2110.13711
    https://arxiv.org/abs/2202.09729
"""

import haiku as hk
import jax.numpy as jnp
from jax.random import PRNGKey
from s4d import S4DEncoder


class S4DMLM(hk.Module):
    def __init__(self, A, L, H, name=None):
        super().__init__(name=name)
        layers = [S4DEncoder(H, A, name=f"block_{i}") for i in range(L)]
        self.cnn = hk.Sequential(layers)
        self.rnn = hk.DeepRNN(layers)

    def __call__(self, u, state=None, **kwargs):
        if state is None:
            return self.cnn(u)
        else:
            return hk.dynamic_unroll(self.rnn, u, state, **kwargs)


@hk.without_apply_rng
@hk.transform
def model(u, state=None, **kwargs):
    A, L, H = 4, 12, 256  # electra-small-discriminator configuration
    return S4DMLM(A, L, H)(u, state, **kwargs)


u = jnp.zeros((1, 512, 256))  # (B, N, D)
params = model.init(PRNGKey(42), u)
print(f"{hk.data_structures.tree_size(params) / 1e6:.3g}M parameters")
y = model.apply(params, u)  # apply convolutional parametrization
pass

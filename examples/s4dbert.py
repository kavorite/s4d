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
import jax
from s4d import DeepS4DNN, S4DEncoder


@hk.without_apply_rng
@hk.transform
def model(u, training=True, state=None):
    A, L, H = 4, 12, 256  # electra-small-discriminator configuration
    backbone = DeepS4DNN([S4DEncoder(H, A) for _ in range(L)])
    if training:
        y = backbone(u)
    else:
        state = backbone.initial_state(u.shape[0])
        y, state = hk.dynamic_unroll(backbone, u, state, time_major=False)
    return y if state is None else (y, state)


rngs = hk.PRNGSequence(42)
inputs = jax.random.normal(next(rngs), [1, 512, 256])  # (B, N, D)
params = model.init(next(rngs), inputs)
print(f"{hk.data_structures.tree_size(params) / 1e6:.3g}M parameters")
y_cnn = model.apply(params, inputs)
y_rnn, _ = model.apply(params, inputs, training=False)
pass

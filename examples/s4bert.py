"""
This example creates a simple, lightweight sequence transduction model backbone
by stacking S4 blocks in a manner not dissimilar from an encoder-only
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
import jax
import jax.numpy as jnp
from einops import rearrange
from jax.random import PRNGKey
from s4d.model import S4D


@hk.without_apply_rng
@hk.transform
def model(u, state=None, is_training=True):
    A, L, H = 4, 12, 256  # electra-small-decoder configuration
    d = u.shape[-1]
    u = hk.Linear(H)(u)
    v = None

    def set_v(u):
        nonlocal v
        v = u.copy()
        return u

    def residual_block():
        s4d = S4D(H, n_ssm=H, channels=A)
        ffn = hk.nets.MLP([H, H], act=jax.nn.silu)

        return [
            set_v,
            s4d.convolutional() if is_training else s4d.recurrent(),
            lambda u: (
                rearrange(u, "... h l d -> ... l (h d)")
                if is_training
                else rearrange(u, "... h d -> ...(h d)")
            ),
            hk.LayerNorm(-1, True, True),
            ffn,
            lambda u: u + v,
            jax.nn.silu,
        ]

    layers = sum((residual_block() for _ in range(L)), [])
    if is_training:
        for f in layers:
            u = f(u)
    else:
        core = hk.DeepRNN(layers)
        if state is None:
            state = core.initial_state(batch_size=u.shape[0])
            u, state = hk.dynamic_unroll(core, u, state, time_major=False)
        else:
            u, state = core(u, state)

    u = hk.Linear(d)(u)
    return u if is_training else (u, state)


u = jnp.zeros((1, 512, 256))  # (B N D)
params = model.init(PRNGKey(42), u)
print(f"{hk.data_structures.tree_size(params) / 1e6:.3g}M parameters")
v = model.apply(params, u)  # apply convolutional parametrization
pass

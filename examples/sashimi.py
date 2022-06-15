from math import ceil, floor

import haiku as hk
import jax
import jax.numpy as jnp
from attr import s
from einops import rearrange
from jax.random import PRNGKey
from s4d import DeepS4DNN, S4DEncoder


def round_up(n, d):
    """
    round a unit count n up to a multiple of a divisor d, ensuring that the
    return value is at least n
    """
    k = max(d, int(n + d / 2) // d * d)
    return k + d if k < n else k


class Resampler(hk.RNNCore):
    def initial_state(self, batch_size=None):
        return batch_size


class DnPool(Resampler):
    def __init__(self, length_factor=0.5, depth_factor=2, name=None):
        super().__init__(name=name)
        assert 0 < length_factor <= 1
        assert depth_factor > 1 and isinstance(depth_factor, int)
        self.depth_factor = depth_factor
        self.length_factor = length_factor

    def __call__(self, u, state=None):
        k = round_up(self.depth_factor / self.length_factor, self.depth_factor)
        if state is None:
            v = rearrange(u, "... (n k) d -> ... n (k d)", k=k)
            d = u.shape[-1] * self.depth_factor
        else:
            v = u
            d = u.shape[-1] // self.depth_factor
        y = hk.Linear(k)(v)
        return y if state is None else (y, state)


class UpPool(Resampler):
    def __init__(self, length_factor=2, depth_factor=0.5, name=None):
        super().__init__(name=name)
        assert 0 < depth_factor <= 1
        assert length_factor > 1 and isinstance(length_factor, int)
        self.depth_factor = depth_factor
        self.length_factor = length_factor

    def __call__(self, u, state=None):
        k = round_up(self.length_factor / self.depth_factor, self.length_factor)
        v = hk.Linear(u.shape[-1] * self.length_factor)(u)
        if state is None:
            y = rearrange(v, "... n (k d) -> ... (n k) d", k=k)
        else:
            y = rearrange(v, "... (n d) -> ... n d")
        return y if state is None else (y, state)


class SaShiMi(hk.Module):
    def __init__(self, input_dim, residual_depths, num_heads, name=None):
        super().__init__(name=name)
        H = input_dim
        A = num_heads
        self.dn_stages = []
        self.up_stages = []

        for stage, depth in enumerate(residual_depths):
            H *= 2
            stack = [
                S4DEncoder(H, A, name=f"stage_{stage}_block_{block}")
                for block in range(depth)
            ]
            block = DeepS4DNN([DnPool()] + stack, name=f"stage_{stage}")
            self.dn_stages.append(block)

        for stage, depth in enumerate(residual_depths):
            stage = len(residual_depths) + stage
            stack = [
                S4DEncoder(H, A, name=f"stage_{stage}_block_{block}")
                for block in range(depth)
            ]
            block = DeepS4DNN(stack + [UpPool()], name=f"stage_{stage}")
            self.up_stages.append(block)
            H //= 2

    def initial_state(self, batch_size=None):
        dn_states = tuple(stage.initial_state(batch_size) for stage in self.dn_stages)
        up_states = tuple(stage.initial_state(batch_size) for stage in self.up_stages)
        return dn_states, up_states

    def __call__(self, u, state=None, timescale=1.0):
        scales = []
        if state is not None:
            dn_states, up_states = map(list, state)

        for i, stage in enumerate(self.dn_stages):
            scales.append(u)
            if state is not None:
                u, dn_states[i] = stage(u, state=dn_states[i], timescale=timescale)
            else:
                u = stage(u, state=None, timescale=timescale)

        for v, stage in zip(scales[::-1], self.up_stages):
            if state is not None:
                u, up_states[i] = stage(u, state=up_states[i], timescale=timescale)
            else:
                u = stage(u, state=None, timescale=timescale)
            u += v  # UNet skip connection

        if state is not None:
            return u, (tuple(dn_states), tuple(up_states))
        else:
            return u


@hk.without_apply_rng
@hk.transform
def model(u, is_training=True):
    backbone = SaShiMi(input_dim=1, residual_depths=[2, 2], num_heads=4)
    if is_training:
        y = backbone(u)
    # TODO: autoregressive inference. Contributions welcome
    # state = backbone.initial_state(u.shape[0])
    # # pass in enough samples at each time-step for a complete forward pass
    # u = rearrange(u, "... (n k) d -> ... n (k d)", k=2 ** len(backbone.dn_stages))
    # y, state = hk.dynamic_unroll(backbone, u, state, time_major=False)
    return y  # if is_training else (y, state)


u = jnp.zeros((1, 1024, 1))  # (B N D)
params = jax.jit(model.init, backend="cpu")(PRNGKey(42), u)
print(f"{hk.data_structures.tree_size(params) / 1e6:.3g}M parameters")
y_cnn = jax.jit(model.apply, backend="cpu")(params, u)
# y_rnn, _ = jax.jit(model.apply, backend="cpu", static_argnums=(2,))(params, u, False)
pass

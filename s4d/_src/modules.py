import haiku as hk
import jax
from einops import rearrange

from .core import S4D, inject_timescale


class DeepS4DNN(hk.RNNCore):
    def __init__(self, layers, name=None):
        super().__init__(name=name)
        self.layers = layers

    def rnn(self):
        return hk.DeepRNN(self.layers)

    def cnn(self):
        return hk.Sequential(self.layers)

    def initial_state(self, batch_size=None):
        return self.rnn().initial_state(batch_size)

    def __call__(self, u, state=None, timescale=1.0):
        with inject_timescale(timescale):
            if state is not None:
                return self.rnn()(u, state)
            else:
                return self.cnn()(u)


class S4DEncoder(hk.RNNCore):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        activation=jax.nn.silu,
        bidirectional=False,
        expand_factor=4,
        norm=True,
        name=None,
    ):
        super().__init__(name=name)
        H, A = hidden_dim, num_heads
        self.s4d = S4D(H, n_ssm=H, channels=A, bidirectional=bidirectional)
        if norm:
            self.nrm = hk.LayerNorm(-1, True, False)
        else:
            self.nrm = lambda x: x
        self.ffn = hk.nets.MLP([H * expand_factor, H], activation=activation)
        self.act = activation

    def initial_state(self, batch_size=None):
        return self.s4d.initial_state(batch_size)

    def __call__(self, u, state=None, timescale=1.0):
        u = self.nrm(u)
        if state is None:
            v = self.s4d.cnn(timescale)(u)
        else:
            v, state = self.s4d.rnn(timescale)(u, state)
        v = rearrange(v, "... h d -> ... (h d)")
        y = self.act(u + self.ffn(self.act(v)))
        return y if state is None else (y, state)

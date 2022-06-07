"dm-haiku port of https://github.com/HazyResearch/state-spaces/blob/main/src/models/sequence/ss/standalone/s4d.py"
import warnings

import haiku as hk
import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from haiku.initializers import RandomNormal, RandomUniform


class S4DCore(hk.RNNCore):
    def __init__(self, dA, dB, dC, D, name=None):
        super().__init__(name=name)
        self.dA = dA
        self.dB = dB
        self.dC = dC
        self.D = D

    def initial_state(self, batch_size=None):
        shape = (batch_size,) if batch_size is not None else ()
        return jnp.zeros(shape + self.dB.shape, dtype=self.dC.dtype)

    def __call__(self, u, state):
        v = jnp.einsum("h n, ... h n -> ... h n", self.dA, state)
        z = jnp.einsum("h n, ... h -> ... h", self.dB, u)
        state = v + z[..., None]
        y = jnp.einsum("c h n, ... h n -> c h", self.dC, state)
        y = y + u[..., None, :] * self.D
        return 2 * y.real, state


class S4DConv(hk.Module):
    def __init__(self, w, C, D, dt, channels, name=None):
        super().__init__(name=name)
        self.w = w
        self.C = C
        self.D = D
        self.dt = dt
        self.channels = channels

    def _kernel(self, l=1):
        h = self.dt.shape[-1]
        k = h // self.w.shape[0]
        w = repeat(self.w, "t n -> (v t) n", v=k)
        C = self.C + 0j
        dtA = w * self.dt[..., None]
        K = jnp.arange(l) * dtA[..., None]
        C = C * ((jnp.exp(dtA) - 1.0) / w)
        K = jnp.einsum("c h n, h n l -> c h l", C, jnp.exp(K))
        K = 2 * K.real
        return K

    def __call__(self, u):
        u = u.swapaxes(-1, -2)
        l = u.shape[-1]
        k = self._kernel(l=l)
        k_f = jnp.fft.rfft(k, n=2 * l - 1)
        u_f = jnp.fft.rfft(u, n=2 * l - 1)
        y_f = jnp.einsum("... h l, c h l -> ... c h l", u_f, k_f)
        y = jnp.fft.irfft(y_f, n=2 * l)[..., :l]
        y = y + jnp.einsum("... h l, c h -> ... c h l", u, self.D)
        return rearrange(y, "... c h l -> ... l c h")


class S4D(hk.Module):
    def __init__(
        self, h, n=64, channels=1, dt_min=0.001, dt_max=0.1, n_ssm=1, name=None
    ):
        super().__init__(name=name)
        if not (n_ssm % h == 0 and n_ssm // h > 0):
            warnings.warn("n_ssm should divide h")
        self.channels = channels
        self.D = hk.get_parameter("D", (self.channels, h), init=RandomNormal())
        self.n_ssm = n_ssm
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.h = h
        self.n = n

    @hk.transparent
    def _w(self):
        h, n = self.n_ssm, self.n
        real = 0.5 * jnp.ones((h, n // 2))
        imag = repeat(jnp.arange(n // 2), "n -> h n", h=h)
        imag = (
            1 / jnp.pi * n * (n / (1 + 2 * imag) - 1)
        )  # based on asymptotics of default HiPPO matrix
        w = -real + 1j * imag
        return repeat(w, "t n -> (v t) n", v=h // w.shape[0])

    @hk.transparent
    def _B(self):
        B = hk.get_parameter(
            "B",
            (self.n_ssm, self.n // 2),
            init=RandomNormal(),
            dtype=jnp.complex64,
        )
        zeta = 2 * jnp.sum((-B * self._w()) ** 2, axis=-1, keepdims=True)
        return B / zeta**0.5

    @hk.transparent
    def _C(self):
        B = self._B()
        B = repeat(B, "t n -> (v t) n", v=self.n_ssm // B.shape[-2])
        C = hk.get_parameter(
            "C",
            (self.channels, self.h, self.n // 2),
            dtype=jnp.complex64,
            init=RandomNormal(),
        ) * repeat(B, "t n -> (v t) n", v=self.h // self.n_ssm)
        return C

    @hk.transparent
    def _dt(self, timescale=1.0):
        floor = jnp.log(self.dt_min)
        ceil = jnp.log(self.dt_max)
        log_dt = (
            hk.get_parameter(
                "log_dt",
                (self.h,),
                init=RandomUniform(self.dt_min, self.dt_max),
            )
            * (ceil - floor)
            + floor
        )
        return jnp.exp(log_dt) * timescale

    def recurrent(self, timescale=1.0):
        dtA = self._w() * self._dt(timescale)[..., None]
        dA = jnp.exp(dtA)
        dC = self._C() * (jnp.exp(dtA - 1.0) / self._w())
        dB = jnp.ones((self.h, self.n // 2))

        return S4DCore(dA, dB, dC, self.D)

    def convolutional(self, timescale=1.0):
        return S4DConv(self._w(), self._C(), self.D, self._dt(timescale), self.channels)

    def __call__(self, u, timescale=1.0, state=None):
        if state is None:
            return self.convolutional(timescale)(u)
        else:
            return self.recurrent(timescale)(u, state)


class S4DEncoder(hk.RNNCore):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        activation=jax.nn.silu,
        name=None,
    ):
        super().__init__(name=name)
        H, A = hidden_dim, num_heads
        self.s4d = S4D(H, n_ssm=H, channels=A)
        self.nrm = hk.LayerNorm(-1, True, True)
        self.pt1 = hk.Linear(hidden_dim)
        self.pt2 = hk.Linear(hidden_dim)
        self.act = activation

    def initial_state(self, batch_size=None):
        return self.s4d.recurrent().initial_state(batch_size)

    def __call__(self, u, state=None, timescale=1.0):
        if state is None:
            v = self.s4d.convolutional(timescale)(u)
        else:
            v, state = self.s4d.recurrent(timescale)(u, state)

        v = rearrange(v, "... h d -> ... (h d)")
        v = self.act(self.pt1(self.nrm(v)))
        y = self.act(self.pt2(u + v))
        return y if state is None else (y, state)

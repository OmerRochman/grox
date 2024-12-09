from abc import abstractmethod
from typing import Callable

import jax
import jax.numpy as jnp

from jax import Array


class BaseSolver:
    r"""Base solver for stochastic differential equations.

    This class serves as a base for implementing numerical solvers for stochastic
    differential equations of the form:
        dX_t = f(X_t, t)dt + g(X_t, t)dW_t

    where f is the drift function and g is the diffusion function.

    Args:
        drift: Function for the drift term f(x,t).
        diffusion: Function for the diffusion term g(x,t).
        ts: Array of time points to solve for.

    Attributes:
        drift: The drift function.
        diffusion: The diffusion function.
        ts: Time points array.
        t0: Initial time.
        t1: Final time.
        dt: Time step size.
        dw: Square root of absolute time step (for Brownian increments).

    Methods:
        step: Abstract method to implement one step of the numerical scheme.
        solve: Solves the SDE, optionally returning full trajectory.
    """

    def __init__(self, drift: Callable, diffusion: Callable, ts: Array):
        self.drift = drift
        self.diffusion = diffusion
        self.ts = ts
        self.t0 = ts[0]
        self.t1 = ts[-1]
        self.dt = ts[1] - ts[0]
        self.dw = jnp.sqrt(jnp.abs(self.dt))

    @abstractmethod
    def step(self, x: Array, t: Array, rng: jax.random.key, *args, **kwargs) -> Array:
        pass

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def solve(
        self, x0: Array, rng: jax.random.key, full_trajectory: bool = False
    ) -> Array:
        r"""Solves the SDE.

        Args:
            x0: Initial condition.
            rng: JAX random key.
            full_trajectory: If True, returns the full trajectory.
        """
        if full_trajectory:

            def body_fn(carry, t):
                x, rng = carry
                rng, step_rng = jax.random.split(rng)
                x = self.step(x, t, step_rng)
                return (x, rng), x

            (xT, _), xs = jax.lax.scan(body_fn, (x0, rng), self.ts)
            return xT, xs

        else:

            def body_fn(i, carry):
                x, rng = carry
                rng, step_rng = jax.random.split(rng)
                x = self.step(x, self.ts[i], step_rng)
                return x, rng

            xT, _ = jax.lax.fori_loop(0, len(self.ts), body_fn, (x0, rng))

            return xT, None


class EulerMaruyama(BaseSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, x: Array, t: Array, rng: jax.random.key, *args, **kwargs) -> Array:
        drift = self.drift(x, t)
        diffusion = self.diffusion(x, t)
        noise = jax.random.normal(rng, x.shape)
        x = x + drift * self.dt + diffusion * self.dw * noise
        return x


class HeunSolver(BaseSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, x: Array, t: Array, *args, **kwargs) -> Array:
        fx = self.drift(x, t)
        xhat = x + fx * self.dt
        that = t + self.dt
        that = jnp.where(that >= 0.0, that, 0.0)
        x = x + 0.5 * (fx + self.drift(xhat, that)) * self.dt
        return x

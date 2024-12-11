from typing import Callable

import jax
from jax import Array


class ReverseSDE:
    def __init__(self, score: Callable, drift: Callable, diffusion: Callable):
        r"""Reverse SDE to match a forward SDE defined by the score, drift, and diffusion functions.

        In the case of a Denoiser, the score has to be computed by the Denoiser using score(x, t) = (denoiser(x, t) -x) / sigma(t) ** 2.

        Args:
            score: Score function.
            drift: Drift function of the forward SDE.
            diffusion: Diffusion function of the forward SDE.
        """
        self.score = score
        self.f = drift
        self.g = diffusion

    def drift(self, x: Array, t: Array) -> Array:
        return self.f(x, t) - self.g(x, t) ** 2 * self.score(x, t)

    def diffusion(self, x: Array, t: Array) -> Array:
        return self.g(x, t)


class ProbODE:
    def __init__(self, score: Callable, drift: Callable, diffusion: Callable):
        """ProbablityODE, and ODE with the same time-marginal distributions as the ReverseSDE.

        In the case of a Denoiser, the score has to be computed by the Denoiser using score(x, t) = (denoiser(x, t) -x) / sigma(t) ** 2.

        Args:
            score: Score function.
            drift: Drift function of the forward SDE.
            diffusion: Diffusion function of the forward SDE.
        """
        self.score = score
        self.f = drift
        self.g = diffusion

    def drift(self, x: Array, t: Array) -> Array:
        return self.f(x, t) - 0.5 * self.g(x, t) ** 2 * self.score(x, t)

    def diffusion(self, x: Array, t: Array) -> Array:
        return 0.0


class EmdODE:
    def __init__(
        self, denoiser: Callable, scale_schedule: Callable, sigma_schedule: Callable
    ):
        r"""A generalization of ProbODE where schedule functions are used instead of the drift and diffusion.

        Reference: Elucidating the Design Space of Diffusion-Based Generative Models, https://arxiv.org/abs/2206.00364

        Args:
            denoiser: Denoiser function.
            scale_schedule: Schedule function for the scale. It is the derivative of the drift.
            sigma_schedule: Schedule function for the sigma. It is the derivative of the diffusion.
        """

        self.denoiser = denoiser
        self.scale_schedule = scale_schedule
        self.sigma_schedule = sigma_schedule

    def drift(self, x: Array, t: Array) -> Array:
        scale, dscale = jax.value_and_grad(self.scale_schedule)(t)
        sigma, dsigma = jax.value_and_grad(self.sigma_schedule)(t)

        term1 = (dsigma / sigma + dscale / scale) * x
        term2 = -scale * dsigma * self.denoiser(x / scale, t) / sigma
        return term1 + term2

    def diffusion(self, x: Array, t: Array) -> Array:
        return 0.0

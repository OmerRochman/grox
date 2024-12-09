from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array


def VP_schedule(
    beta_min: float, beta_max: float, **kwargs
) -> tuple[Callable, Callable]:
    r"""Variance preserving schedule.

    Args:
        beta_min: min noise level.
        beta_max: max noise level.

    Returns:
        scale_schedule: Function that gives the desired signal strength as a function of t.
        sigma_schedule: Function that gives the desired noise level as a function of t.
    """

    def scale_schedule(t) -> Array:
        log_mean_coeff = 0.5 * (beta_max - beta_min) * t**2 + beta_min * t
        scale = jax.lax.rsqrt(jnp.exp(log_mean_coeff) + 1e-6)
        return scale

    def sigma_schedule(t: Array) -> Array:
        log_mean_coeff = 0.5 * (beta_max - beta_min) * t**2 + beta_min * t
        sigma = jnp.sqrt(jnp.exp(log_mean_coeff) - 1 + 1e-6)
        return sigma

    return scale_schedule, sigma_schedule


def VE_schedule(
    sigma_min: float = 0.02, sigma_max: float = 100.0, **kwargs
) -> tuple[Callable, Callable]:
    r"""Variance exploding schedule.

    Args:
        sigma_min: min noise level.
        sigma_max: max noise level.

    Returns:
        scale_schedule: Function that gives the desired signal strength as a function of t.
        sigma_schedule: Function that gives the desired noise level as a function of t.
    """
    # TODO: implement


def EDM_schedule():
    r"""
    Noise schedule from Elucidating the Design Space of Diffusion-Based Generative Models

    Reference: Elucidating the Design Space of Diffusion-Based Generative Models, https://arxiv.org/abs/2206.00364
    """

    # TODO: implement
    pass

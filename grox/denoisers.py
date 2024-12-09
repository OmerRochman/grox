from abc import abstractmethod
from typing import Callable
from jax import Array
import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange


class Denoiser:
    r"""Abstract class for denoising functions.
    Args:
        backbone: Neural backbone.
        scale_schedule: Schedule function that give the signal scale (:math:`\alpha_t`) at time :math:`t`.
        sigma_schedule: Schedule function that give the noise scale at time :math:`t`.
    """

    def __init__(
        self, backbone: nn.Module, scale_schedule: Callable, sigma_schedule: Callable
    ):
        self.backbone = backbone
        self.scale_schedule = scale_schedule
        self.sigma_schedule = sigma_schedule

    @abstractmethod
    def __call__(self, params, x: Array, t: Array, **kwargs) -> Array:
        r"""Forward of denoiser

        Args:
            params (pytree): Flax module parameters
            x: Input tensor of any shape
            t: Time tensor matching the first dimension of x (batch size).
        """
        raise NotImplementedError


class VPDenoiser(Denoiser):
    def __call__(self, params, x: Array, t: Array, **kwargs) -> Array:
        t = rearrange(t, "... -> ..." + " 1" * (x.ndim - 1))
        sigma = self.sigma_schedule(t)
        c_in = 1 / jnp.sqrt(sigma**2 + 1)
        c_out = -sigma
        c_skip = jnp.ones_like(x)
        c_noise = t

        x_out = self.backbone.apply(params, c_in * x, c_noise.squeeze())
        return c_skip * x + c_out * x_out


class VEDenoiser(Denoiser):
    def __call__(self, params, x: Array, t: Array, **kwargs) -> Array:
        t = rearrange(t, "... -> ..." + " 1" * (x.ndim - 1))
        sigma = self.sigma_schedule(t)
        c_in = 1
        c_out = sigma
        c_skip = 1
        c_noise = jnp.log(0.5 * sigma)
        out = self.backbone.apply(params, c_in * x, c_noise.squeeze())
        out = c_skip * x + c_out * out
        return out

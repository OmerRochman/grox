from typing import Callable, List

import jax.lax as lax
import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn
from jax import Array


class Modulation(nn.Module):
    r"""Time modulation layer.

    Args:
        time_embed_features: Number of features in the time embedding.
        input_features: Number of input features of the layer being modulated.
        output_features: Number of output features of the layer being modulated.
        spatial_dims: Number of spatial dimensions. Default: 2.
    """

    time_embed_features: int
    input_features: int
    output_features: int
    spatial_dims: int = 2

    def setup(self):
        self.linear1 = nn.Dense(self.time_embed_features)
        self.linear2 = nn.Dense(2 * self.input_features + self.output_features)
        self.activation = nn.swish

    def __call__(self, t: Array) -> tuple[Array, Array, Array]:
        r"""

        Args:
            t: Input array of shape (B,)
        """
        x = self.linear1(t)
        x = self.activation(x)
        modulation_params = self.linear2(x)
        a, b, c = jnp.split(
            modulation_params, (self.input_features, 2 * self.input_features), axis=-1
        )

        a = rearrange(a, "... d -> ..." + " 1" * self.spatial_dims + " d")
        b = rearrange(b, "... d -> ..." + " 1" * self.spatial_dims + " d")
        c = rearrange(c, "... d -> ..." + " 1" * self.spatial_dims + " d")

        return a, b, c


class SineEncoding(nn.Module):
    r"""Sine encoding for temporal features

    Args:
        features: Number of features to encode.
        omega: Frequency of the sine encoding.
        time_embed_features: Number of features in the time embedding.
        act: Activation function.
    """

    features: int
    omega: float = 1e4
    time_embed_features: int = 64
    act: Callable = nn.swish

    def setup(self):
        assert self.features % 2 == 0, "Features must be even."
        freqs = jnp.linspace(0, 1, self.features // 2, dtype=jnp.float32)
        freqs = self.omega ** (-freqs)
        self.freqs = freqs
        self.linear1 = nn.Dense(self.time_embed_features)
        self.linear2 = nn.Dense(self.time_embed_features)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        r"""

        Args:
            x: Input array of shape (B,)
        """
        x = x[..., None]
        sin_component = jnp.sin(self.freqs * x)
        cos_component = jnp.cos(self.freqs * x)
        x = jnp.concatenate((sin_component, cos_component), axis=-1)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class MLP(nn.Module):
    r"""Simple MLP

    Args:
        num_inputs: Number of input features.
        num_outputs: Number of output features.
        architecture: List of hidden layer sizes.
        activation: Activation function
    """

    num_inputs: int
    num_outputs: int
    architecture: List[int]
    activation: Callable = nn.swish

    def setup(self):
        self.layers = [
            nn.Dense(size) for size in list(self.architecture) + [self.num_outputs]
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        return self.layers[-1](x)


class ResBlock(nn.Module):
    r"""Residual block with modulation

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        kernel_size: Kernel size.
        strides: Strides.
        act: Activation function.
        emb_features: Number of features in the time embedding.
        dropout_rate: Dropout rate. TODO: Implement dropout.
        padding: Padding type, "SAME" or "CIRCULAR".

    """

    in_features: int
    out_features: int
    kernel_size: tuple = (3, 3)
    strides: tuple = (1, 1)
    act: Callable = nn.swish
    emb_features: int = 64
    dropout_rate: float = 0.0
    padding: str = "SAME"

    def setup(self):
        # self.dropout = nn.Dropout(self.dropout_rate)
        self.conv1 = nn.Conv(
            self.out_features,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
        )
        self.conv2 = nn.Conv(
            self.out_features,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
        )
        self.modulation = Modulation(
            self.emb_features, self.in_features, self.out_features
        )
        self.norm = nn.LayerNorm()

    def __call__(self, x: Array, t: Array):
        a, b, c = self.modulation(t)
        y = self.norm(x)
        y = (a + 1) * x + b
        y = self.conv1(y)
        y = self.act(y)
        # y = self.dropout(y)
        y = self.conv2(y)
        y = x + c * y
        return y * lax.rsqrt(1 + c**2)


class UNet2D(nn.Module):
    r"""2D U-Net with time modulation and sine encoding.

    Args:
        architecture: List of hidden layer sizes.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size.
        activation: Activation function.
        dropout_rate: Dropout rate.
        emb_features: Number of features in the time embedding.
        padding: Padding type, "SAME" or "CIRCULAR". Circular padding is used for periodic boundary conditions in 1D problems, where H is time and W is space.
    """

    architecture: List[int]
    in_channels: int
    out_channels: int
    kernel_size: tuple = (3, 3)
    activation: Callable = nn.swish
    dropout_rate: float = 0.0
    emb_features: int = 64
    padding: str = "SAME"

    def setup(self):
        self.time_embedding = SineEncoding(
            features=64, time_embed_features=self.emb_features
        )
        self.encoder = MLP(
            self.in_channels, self.architecture[0], [64, 64, 64], self.activation
        )
        self.decoder = MLP(
            self.architecture[0], self.out_channels, [64, 64, 64], self.activation
        )

        down_blocks = []
        for i in range(len(self.architecture) - 1):
            down_blocks.append(
                ResBlock(
                    self.architecture[i],
                    self.architecture[i],
                    kernel_size=self.kernel_size,
                    act=self.activation,
                    emb_features=self.emb_features,
                    dropout_rate=self.dropout_rate,
                    padding=self.padding,
                )
            )
            down_blocks.append(
                nn.Sequential(
                    [
                        nn.LayerNorm(),
                        nn.Conv(
                            self.architecture[i + 1],
                            self.kernel_size,
                            strides=(2, 2),
                            padding=self.padding,
                        ),
                    ]
                )
            )
        self.down_blocks = down_blocks

        up_blocks = []
        for i in range(len(self.architecture) - 2, -1, -1):
            up_blocks.append(
                ResBlock(
                    self.architecture[i + 1],
                    self.architecture[i + 1],
                    kernel_size=self.kernel_size,
                    act=self.activation,
                    emb_features=self.emb_features,
                    dropout_rate=self.dropout_rate,
                    padding=self.padding,
                )
            )
            up_blocks.append(
                nn.Sequential(
                    [
                        nn.LayerNorm(),
                        nn.ConvTranspose(
                            self.architecture[i],
                            self.kernel_size,
                            strides=(2, 2),
                            padding=self.padding,
                        ),
                    ]
                )
            )
        self.up_blocks = up_blocks

        self.bottleneck = nn.Conv(
            self.architecture[-1],
            self.kernel_size,
            strides=(1, 1),
            padding=self.padding,
        )

    def __call__(self, x: Array, t: Array) -> Array:
        r"""Channel last forward pass.

        Args:
            x: Input array of shape (B, H, W, C).
            t: Time array of shape (B,).
        """
        x = self.encoder(x)
        if self.padding == "CIRCULAR":
            k = self.kernel_size[0] - 1
            zeros = jnp.zeros_like(x[..., :k, :, :])
            x = jnp.concatenate([zeros, x, zeros], axis=-3)
        t = self.time_embedding(t)

        skip_connections = [x]
        for down_block in self.down_blocks:
            if isinstance(down_block, ResBlock):
                x = down_block(x, t)
            else:
                x = down_block(x)
                skip_connections.append(x)

        x = self.bottleneck(x)
        x = self.activation(x)

        skip = skip_connections.pop(-1)
        for up_block in self.up_blocks:
            if isinstance(up_block, ResBlock):
                x = up_block(x, t) + skip
                skip = skip_connections.pop(-1)
            else:
                x = up_block(x)

        if self.padding == "CIRCULAR":
            x = x[..., k:-k, :, :]
        x = self.decoder(x)
        return x

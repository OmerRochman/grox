from typing import Callable, List

from jax.image import resize
import jax.lax as lax
import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn
from jax import Array


def small_init(key, shape, dtype=jnp.float32):
    default_kernel_init = nn.initializers.lecun_normal()
    return default_kernel_init(key, shape, dtype) * 0.01


class Modulation(nn.Module):
    r"""Time modulation layer.

    Args:
        time_embed_features: Number of features in the time embedding.
        input_features: Number of input features of the layer being modulated.
        output_features: Number of output features of the layer being modulated.
        spatial_dims: Number of spatial dimensions. Default: 2.
    """

    input_features: int
    output_features: int
    spatial_dims: int = 2

    def setup(self):
        self.linear1 = nn.Dense(2 * self.input_features + self.output_features)
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
        n_freqs: Number of frequencies to encode.
        omega: Frequency of the sine encoding.
        time_embed_features: Number of features in the time embedding.
        act: Activation function.
    """

    n_freqs: int
    omega: float = 1e4
    time_embed_features: int = 64
    act: Callable = nn.swish

    def setup(self):
        assert self.n_freqs % 2 == 0, "Features must be even."
        freqs = jnp.linspace(0, 1, self.n_freqs // 2, dtype=jnp.float32)
        self.freqs = self.omega ** (-freqs)
        self.freqs = self.freqs[None, :]
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
        num_outputs: Number of output features.
        architecture: List of hidden layer sizes.
        activation: Activation function
    """

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
    r"""Residual block with modulation, and the capacity to change the hidden dimension.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        kernel_size: Kernel size.
        strides: Strides.
        act: Activation function.
        dropout_rate: Dropout rate. TODO: Implement dropout.
        padding: Padding type, "SAME" or "CIRCULAR".

    """

    in_features: int
    out_features: int
    kernel_size: tuple = (3, 3)
    strides: tuple = (1, 1)
    act: Callable = nn.swish
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
        self.modulation = Modulation(self.in_features, self.out_features)
        self.norm = nn.LayerNorm()

    def __call__(self, x: Array, t: Array, train: bool = True) -> Array:
        a, b, c = self.modulation(t)
        y = (a + 1) * x + b
        y = self.norm(y)

        y = self.conv1(y)
        y = self.act(y)
        # y = self.dropout(y, deterministic=False)
        y = self.conv2(y)
        if x.shape[-1] == y.shape[-1]:
            y = x + c * y
        else:
            y = y * c

        return y * lax.rsqrt(1 + c**2)


class Resize(nn.Module):
    def setup(self):
        self.norm = nn.LayerNorm()

    def __call__(
        self, x: Array, shape: tuple, method: str = "linear", antialias: bool = True
    ) -> Array:
        x = self.norm(x)
        return resize(image=x, shape=shape, method=method, antialias=antialias)


class UNet(nn.Module):
    r"""2D U-Net with time modulation and sine encoding.

    Args:
        out_channels: Number of output channels.
        blocks_per_layer: Number of ResBlocks per layer.
        hidden_channels_per_layer: Number of hidden channels in each layer.
        kernel_size: Kernel size.
        activation: Activation function.
        dropout_rate: Dropout rate.
        emb_features: Number of features in the time embedding.
        n_freqs: Number of frequencies in the time embedding.
        padding: Padding type, "SAME" or "CIRCULAR". Circular padding is used for periodic boundary conditions in 1D problems, where H is time and W is space.
    """

    out_channels: int
    blocks_per_layer: tuple = (2, 2)
    hidden_channels_per_layer: tuple = (64, 64)
    kernel_size: tuple = (3, 3)
    activation: Callable = nn.swish
    dropout_rate: float = 0.0
    emb_features: int = 64
    n_freqs: int = 64
    padding: str = "SAME"
    omega: float = 1e4
    mlp_hiddden_channels: tuple = (64, 64, 64)

    def setup(self):
        assert len(self.blocks_per_layer) == len(self.hidden_channels_per_layer)

        self.time_embedding = SineEncoding(
            n_freqs=64, time_embed_features=self.emb_features, omega=self.omega
        )
        self.encoder = MLP(
            self.hidden_channels_per_layer[0],
            self.mlp_hiddden_channels,
            self.activation,
        )
        self.decoder = MLP(
            self.out_channels,
            self.mlp_hiddden_channels,
            self.activation,
        )

        down_blocks = []
        up_blocks = []
        prev_channels = self.hidden_channels_per_layer[0]
        for num_blocks, channels in zip(
            self.blocks_per_layer, self.hidden_channels_per_layer
        ):
            for i in range(num_blocks):
                down_blocks.append(
                    ResBlock(
                        channels if i > 0 else prev_channels,
                        channels,
                        kernel_size=self.kernel_size,
                        act=self.activation,
                        dropout_rate=self.dropout_rate,
                        padding=self.padding,
                    )
                )
            down_blocks.append(
                nn.Sequential(
                    [
                        nn.LayerNorm(),
                        nn.Conv(
                            channels,
                            self.kernel_size,
                            strides=(2, 2),
                            padding=self.padding,
                        ),
                    ]
                )
            )
            for i in range(num_blocks):
                up_blocks.append(
                    ResBlock(
                        channels,
                        channels,
                        kernel_size=self.kernel_size,
                        act=self.activation,
                        dropout_rate=self.dropout_rate,
                        padding=self.padding,
                    )
                )

            up_blocks.append(
                nn.Sequential(
                    [
                        nn.LayerNorm(),
                        nn.ConvTranspose(
                            channels,
                            self.kernel_size,
                            strides=(2, 2),
                            padding=self.padding,
                            kernel_init=small_init,
                        ),
                        # Resize(),
                    ]
                )
            )
            prev_channels = channels

        self.down_blocks = down_blocks
        self.up_blocks = up_blocks

        self.bottleneck = nn.Conv(
            self.hidden_channels_per_layer[-1],
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

        skip_connections = []
        for down_block in self.down_blocks:
            if isinstance(down_block, ResBlock):
                x = down_block(x, t)
            else:
                skip_connections.append(x)
                x = down_block(x)

        x = self.bottleneck(x)
        x = self.activation(x)

        for up_block in self.up_blocks[::-1]:
            if isinstance(up_block, ResBlock):
                x = up_block(x, t)
            else:
                skip = skip_connections.pop(-1)
                # shape = skip.shape
                x = up_block(x) + skip

        if self.padding == "CIRCULAR":
            x = x[..., k:-k, :, :]
        x = self.decoder(x)
        return x

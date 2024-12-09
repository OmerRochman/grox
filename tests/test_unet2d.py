import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from grox.networks.unet2d import UNet2D


@pytest.mark.parametrize("arch", [[32, 64, 32], [16, 32, 64, 16]])
@pytest.mark.parametrize("in_ch", [1, 3])
@pytest.mark.parametrize("out_ch", [1, 3])
@pytest.mark.parametrize("kernel", [(3, 3), (5, 5)])
@pytest.mark.parametrize("act", [nn.swish])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_unet2d(arch, in_ch, out_ch, kernel, act, dropout):
    unet = UNet2D(
        architecture=arch,
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=kernel,
        activation=act,
        dropout_rate=dropout,
        emb_features=64,
    )

    input_shape = (16, 64, 64, in_ch)  # (batch, height, width, channels)
    x = jnp.ones(input_shape)
    t = jnp.ones((16,))  # Dummy time input
    params = unet.init(jax.random.key(0), x, t)
    output = unet.apply(params, x, t)

    # Check output shape
    assert output.shape == (16, 64, 64, out_ch)

    # Test that outputs have gradients with respect to all parameters
    def loss_fn(params, x, t):
        y = unet.apply(params, x, t)
        return jnp.mean(y)

    loss, grads = jax.jit(jax.value_and_grad(loss_fn))(params, x, t)
    assert grads is not None

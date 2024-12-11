import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from grox.networks.unet import UNet


@pytest.mark.parametrize("hid_ch", [[32, 64], [16, 16]])
@pytest.mark.parametrize("bl_lyr", [[1, 3], [2, 2]])
@pytest.mark.parametrize("out_ch", [1, 3])
@pytest.mark.parametrize("kernel", [(3, 3), (5, 5)])
@pytest.mark.parametrize("act", [nn.swish])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_unet2d(hid_ch, bl_lyr, out_ch, kernel, act, dropout):
    unet = UNet(
        hidden_channels_per_layer=hid_ch,
        blocks_per_layer=bl_lyr,
        out_channels=out_ch,
        kernel_size=kernel,
        activation=act,
        dropout_rate=dropout,
        emb_features=64,
    )

    input_shape = (16, 64, 64, 1)  # (batch, height, width, channels)
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

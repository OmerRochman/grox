import jax.numpy as jnp
import jax.random as random
import pytest

from grox.solvers import EulerMaruyama


@pytest.mark.parametrize("x0", [jnp.array([1.0]), jnp.array([[0.5], [1.5]])])
@pytest.mark.parametrize("ts", [jnp.linspace(0, 1, 10), jnp.linspace(0, 2, 20)])
def test_euler_maruyama(x0, ts):
    """Test nans and infs are not produced by the Euler-Maruyama method."""

    def drift(x, t):
        return -x

    def diffusion(x, t):
        return jnp.ones_like(x) * 1.0 * (1 - t)

    rng = random.PRNGKey(0)
    solver = EulerMaruyama(drift, diffusion, ts)
    xT, xs = solver.solve(x0, rng, full_trajectory=True)

    assert xs.shape == (len(ts),) + x0.shape
    assert xT.shape == x0.shape
    assert jnp.all(jnp.isfinite(xs))


@pytest.mark.parametrize("X0", [1.0, -3.5])
@pytest.mark.parametrize("mu", [0.05, 1.1])
@pytest.mark.parametrize("sigma", [0.2, 1.3])
def test_euler_maruyama_gbm(X0, mu, sigma):
    """Test for the convergence of the Euler-Maruyama method for the Geometric Brownian Motion https://en.wikipedia.org/wiki/Geometric_Brownian_motion."""
    T = 1.0
    N = 50_000
    steps = 50_000
    ts = jnp.linspace(0, 1, steps)

    def drift(x, t):
        return mu * x

    def diffusion(x, t):
        return sigma * x

    solver = EulerMaruyama(drift, diffusion, ts)
    rng = random.PRNGKey(0)

    rng, subkey = random.split(rng)
    X_T, _ = solver.solve(jnp.stack([X0] * N), subkey)

    sample_mean = jnp.mean(X_T)
    sample_var = jnp.var(X_T)

    expected_mean = X0 * jnp.exp(mu * T)
    expected_var = X0**2 * jnp.exp(2 * mu * T) * (jnp.exp(sigma**2 * T) - 1)

    relative_error_mean = jnp.abs(sample_mean - expected_mean) / expected_mean
    relative_error_var = jnp.abs(sample_var - expected_var) / expected_var

    tolerance = 0.01  # 1% tolerance

    assert (
        relative_error_mean < tolerance
    ), f"Mean relative error {relative_error_mean:.4f} exceeds tolerance"
    assert (
        relative_error_var < tolerance
    ), f"Variance relative error {relative_error_var:.4f} exceeds tolerance"

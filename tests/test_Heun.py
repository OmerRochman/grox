from grox.solvers import HeunSolver
import jax
import jax.numpy as jnp


def test_heun_solver_exponential_growth():
    """Test the Heun solver for the exponential growth ODE dy/dt = a*y."""
    a = 1.0
    y0 = 1.0
    t0 = 0.0
    t1 = 1.0
    N = 500
    ts = jnp.linspace(t0, t1, N)

    def drift(x, t):
        return a * x

    diffusion = None

    x0 = jnp.array([y0])

    rng = jax.random.key(0)

    solver = HeunSolver(drift=drift, diffusion=diffusion, ts=ts)

    _, xs = solver.solve(x0, rng, full_trajectory=True)

    numerical_solution = xs[:, 0]

    exact_solution = y0 * jnp.exp(a * ts)

    error = jnp.max(jnp.abs(numerical_solution - exact_solution))

    # Tolerance
    tol = 1e-2

    assert error < tol, f"Maximum error {error} exceeds tolerance {tol}"

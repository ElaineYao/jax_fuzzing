import jax
import jax.numpy as jnp
from jax import grad
from math import ceil

def f(x):
    return jnp.exp(x) / (1 + jnp.exp(x))  # or jax.nn.sigmoid(x)

def integrate_f_0_1(a):

    x = jnp.linspace(0, a, 100*jnp.ceil(a).astype(int))
    y = f(x)
    dx = (x[-1] - x[0]) / (x.size - 1)
    return dx * (0.5 * y[0] + jnp.sum(y[1:-1]) + 0.5 * y[-1])

gradient = grad(integrate_f_0_1)
print(gradient(100.0))
print(f(100.0))
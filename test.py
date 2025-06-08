import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten

def inner_prod(pytree1, pytree2):
    """
    Compute the inner product between two pytrees of matching structure.

    Each leaf should be a scalar or array. The result is a scalar.
    """
    flat1, _ = tree_flatten(pytree1)
    flat2, _ = tree_flatten(pytree2)

    return sum(jnp.vdot(x, y) for x, y in zip(flat1, flat2))

def f_add(a, b):
    return a+b

tangents = (0.3, 1.5)
cotangent = 2.0

primals = (1.0, 2.0)

jvp_primals, tangent_jvp = jax.jvp(f_add, primals, tangents)
print(jvp_primals, tangent_jvp)

primals, f_vjp =jax.vjp(f_add, *primals)

cotangent_vjp = f_vjp(cotangent)
print(cotangent_vjp)

# jvp_res = inner_prod(tangent_jvp, cotangents)
# vjp_res = inner_prod(tangents, cotangent_vjp)

# print(jvp_res, vjp_res)









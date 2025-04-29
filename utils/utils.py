import yaml
import numpy as np
import os, sys

import jax
from jax import lax


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# TODO: add dtype
def rand_like(rng, x):
    shape = np.shape(x)
  
    randn = lambda: np.asarray(rng.randn(*shape))
    results = randn()
    return results


# TODO: Add safe subtraction with the inf-inf=0 semantics
def numerical_jvp(f, primals, tangents, eps):
    # Handle tuple inputs by mapping over them
    delta = jax.tree.map(lambda t: lax.mul(t, eps), tangents)
    f_pos = f(*jax.tree.map(lambda p, d: lax.add(p, d), primals, delta))
    f_neg = f(*jax.tree.map(lambda p, d: lax.sub(p, d), primals, delta))
    return lax.mul((f_pos - f_neg), 0.5 / eps)

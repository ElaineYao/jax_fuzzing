import yaml
import numpy as np
import os, sys
import hashlib
import pandas as pd
import itertools
import jax
from jax import lax
from jax.tree_util import tree_flatten
import jax.numpy as jnp
from functools import partial

def inner_prod(pytree1, pytree2):
    """
    Compute the inner product between two pytrees of matching structure.

    Each leaf should be a scalar or array. The result is a scalar.
    """
    flat1, _ = tree_flatten(pytree1)
    flat2, _ = tree_flatten(pytree2)

    return sum(jnp.vdot(x, y) for x, y in zip(flat1, flat2))

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# TODO: add dtype
def rand_like(rng, x):
    shape = np.shape(x)
  
    randn = lambda: np.asarray(rng.randn(*shape))
    results = randn()
    return results




def generate_hash(*args):
    hasher = hashlib.sha256()
    for arg in args:
        hasher.update(str(arg).encode())
    return hasher.hexdigest()

def param_to_jobname(params):
    parts = []
    for k, v in params.items():
        if isinstance(v, tuple):
            v_str = "_".join(map(str, v))
        else:
            v_str = str(v).replace(".", "p")
        parts.append(f"{k}-{v_str}")
    return "_".join(parts)

def aggregate_results(param_grid, results_dir, output_csv):

    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys,v)) for v in itertools.product(*values)]

    data = []

    for params in param_combinations:
        job_name = param_to_jobname(params)
        result_file = os.path.join(results_dir, f"results_{job_name}.txt")

        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                failure_rate = float(f.read().strip())
            row = params.copy()
            row["failure_rate"] = failure_rate
            data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

def _rand_dtype(rand, shape, dtype, scale=1.0):
    if len(shape) >0 and isinstance(shape[0], tuple):
        return tuple(np.asarray(scale * rand(s), dtype) for s in shape)
    else:
        return np.asarray(scale * rand(shape), dtype)

def rand_default(rng, scale=3):
    return partial(_rand_dtype, rng.standard_normal, scale=scale)

def rand_normal(rng, scale=1):
    return partial(_rand_dtype, rng.standard_normal, scale=scale)

if __name__ == "__main__":
    rng = np.random.default_rng()
    f = rand_default(rng, scale=3)

    print(f(shape=((3,), (3,)), dtype=np.float32))
                

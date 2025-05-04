import yaml
import numpy as np
import os, sys
import hashlib
import pandas as pd
import itertools
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


                

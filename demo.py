import jax
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import itertools
from functools import partial
from utils import utils
from core.oracle import check_grads_ND
from jax import tree_util
import itertools
import pandas as pd

EPS= 1e-4
def f_add(a, b):
    return a+b

def f_sub(a, b):
    return a-b

def f_mul(a, b):
    return a*b

def f_div(a, b):
    return a/b

compatible_shapes = [[(3,)],
[(), (3, 4), (3, 1), (1, 4)],
[(2, 3, 4), (2, 1, 4)]]

def get_shape(nargs):
    shapes=[
        shapes for shape_group in compatible_shapes
        for shapes in itertools.combinations_with_replacement(shape_group, nargs)
        ]
    return shapes


def profile(f, nargs, shape, dtype,mode,order, eps=EPS):
    diff_list = []

    for i in range(10000):
  
        rng = utils.rand_default(np.random.default_rng())
        arg_shape = tuple([shape] * nargs)         # e.g., ((3,), (3,))
    
        args = rng(shape=arg_shape, dtype=dtype)
    
        tangents_rng = utils.rand_normal(np.random.default_rng())
        tangents = tangents_rng(shape=arg_shape, dtype=dtype)
    
        f_output = f(*args)
        cotangent = tangents_rng(shape=f_output.shape, dtype=dtype)


        res_AD, res_ND = check_grads_ND(f, args, cotangent, tangents, order, mode, eps)
        tangent_diff = np.abs(res_AD - res_ND)
        avg_diff = np.mean(tangent_diff)
        min_diff = np.min(tangent_diff)
        max_diff = np.max(tangent_diff)
        diff_list.append((avg_diff, min_diff, max_diff))
    
    return diff_list

def main():
    result_root_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(result_root_dir, exist_ok=True)
    output_csv = os.path.join(result_root_dir, "mean_diff.csv")
    param_grid = {
        "f": [f_add, f_sub, f_mul, f_div],
        "dtype": [np.float32, np.float64],
        "order": [1, 2],
        "eps": [EPS],
        "mode": ["fwd", "rev"],
        "shape": [(3,),(), (3, 4), (3, 1), (1, 4),(2, 3, 4), (2, 1, 4)]
    }

    param_keys = list(param_grid.keys())
    param_combinations = list(itertools.product(*param_grid.values()))
    rows = []
    
    for config in param_combinations:

        config_dict = dict(zip(param_keys, config))
        f = config_dict["f"]
        dtype = config_dict["dtype"]
        order = config_dict["order"]
        eps = config_dict["eps"]
        mode = config_dict["mode"]
        shape = config_dict["shape"]

        nargs = 2
        
        diff_list = profile(f, nargs, shape, dtype, mode, order, eps)
        mean_diff = np.mean([diff[0] for diff in diff_list])
        min_diff = np.min([diff[1] for diff in diff_list])
        max_diff = np.max([diff[2] for diff in diff_list])
        row = {
            "f": f.__name__,
            "dtype": dtype.__name__,
            "order": order,
            "eps": eps,
            "mode": mode,
            "shape": shape,
            "mean_diff": mean_diff,
            "min_diff": min_diff,
            "max_diff": max_diff
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
      
if __name__ == "__main__":
    # diff_list = profile(f_add, 2,(3,), np.float32,"rev",1, eps=EPS)
    # print(np.mean(diff_list))
    main()
    
    
    
        








import jax
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import itertools
from functools import partial
from utils.utils import inner_prod
from core.oracle import check_grads_ND
from jax import tree_util
EPS= 1e-4
def f_add(a, b):
    return a+b

def f_sub(a, b):
    return a-b

def f_mul(a, b):
    return a*b

def f_div(a, b):
    return a/b



def profile(f, dtype,mode,order, eps=EPS):
    diff_list = []

    for i in range(1):
        rng = np.random.default_rng()

        scale = 3
        primal_1 = scale * np.asarray(rng.standard_normal(), dtype=dtype)
        primal_2 = scale * np.asarray(rng.standard_normal(), dtype=dtype)

        primals = (primal_1, primal_2)
        
        f_output = f(*primals)
        print(f_output)

        # res_AD, res_ND = check_grads_ND(f, primals, cotangent, tangents, order, mode, eps)
        # tangent_diff = np.abs(res_AD, res_ND)
       
    
    return diff_list

def main():
    result_root_dir = os.path.join(os.path.dirname(__file__), "results")

    if not os.path.exists(result_root_dir):
        os.makedirs(result_root_dir)
    param_grid = {
        "op": [f_add, f_sub, f_mul, f_div],
        "dtype": [np.float32, np.float64],
        "order": [1, 2],
        "eps": [EPS],
        "mode": ["fwd", "rev"]
    }
    param_combinations = list(itertools.product(*param_grid.values()))
    
    for config in param_combinations:
    
        f = config["f"]
        dtype = config["dtype"]
        order = config["order"]
        eps = config["eps"]
        mode = config["mode"]
        
        diff_list = profile(f, dtype, mode, order, eps)

      
if __name__ == "__main__":
    profile(f_add, np.float32,"fwd",1, eps=EPS)

    
    
        








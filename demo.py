import jax
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import itertools
from functools import partial
from utils.utils import inner_prod
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


def numerical_jvp(f, primals, tangents, eps=EPS):
    
    delta = tuple(t * eps for t in tangents)
    f_pos = f(*(p + d for p, d in zip(primals, delta)))
    f_neg = f(*(p - d for p, d in zip(primals, delta)))
    numerical_jvp_res = tree_util.tree_map(lambda a, b: (a-b)/(2*eps), f_pos, f_neg)
    
    return numerical_jvp_res



def check_gradsND(f, dtype, primals, mode, order=1, eps=1e-4, tangents=None):
    rng = np.random.default_rng()
    eps = np.asarray(eps, dtype=dtype)
    
    if tangents is None:
        tangents = tuple(np.asarray(rng.standard_normal(), dtype=dtype) for _ in primals)


    if mode == "fwd":
        if order == 1:
            _, jvp_AD = jax.jvp(f, primals, tangents)
            jvp_ND = numerical_jvp(f, primals, tangents, eps=eps)
            return jvp_AD, jvp_ND

        def higher_order_jvp(*x):
            _, jvp_f = jax.jvp(f, x, tangents)
            return jvp_f

        return check_gradsND(higher_order_jvp, dtype, primals, mode="fwd",
                             order=order - 1, eps=eps, tangents=tangents)

    elif mode == "rev":
        if order == 1:
            v_out, f_vjp = jax.vjp(f, *primals)
            cotangent = tree_util.tree_map(lambda x: np.asarray(rng.standard_normal(x.shape), dtype=dtype), v_out)
            cotangent_out = f_vjp(cotangent)
            vjp_AD = inner_prod(tangents, cotangent_out)
            tangent_out = numerical_jvp(f, primals, tangents, eps=eps)
            vjp_ND = inner_prod(tangent_out, cotangent)
            return vjp_AD, vjp_ND

        def higher_order_vjp(*x):
            v_out, f_vjp = jax.vjp(f, *x)
            return f_vjp(v_out)
        

        return check_gradsND(higher_order_vjp, dtype, primals, mode="rev",
                             order=order - 1, eps=eps, tangents=tangents)


def profile(f, dtype, atol, rtol, mode,order=1, eps=EPS):
    tangent_diff_list = []
    max_diff_list = []
    portion_list = []

    for i in range(10000):
        rng = np.random.default_rng()

        scale = 3
        primal_1 = scale * np.asarray(rng.standard_normal(), dtype=dtype)
        primal_2 = scale * np.asarray(rng.standard_normal(), dtype=dtype)

        primals = (primal_1, primal_2)
        tangents, tangents_numerical = check_gradsND(f, dtype, primals, mode, order, EPS)
        tangent_diff = np.abs(tangents - tangents_numerical)
       
        max_diff = atol +rtol*np.abs(tangents)
        portion = tangent_diff/max_diff
        max_diff_list.append(max_diff)
        portion_list.append(portion)
        tangent_diff_list.append(tangent_diff)
    return tangent_diff_list, max_diff_list, portion_list

def main():
    result_root_dir = os.path.join(os.path.dirname(__file__), "results")

    if not os.path.exists(result_root_dir):
        os.makedirs(result_root_dir)
    result_dir = os.path.join(result_root_dir, "add")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    param_combinations = [
        {"f": f_add, "dtype": np.float32, "order": 1, "atol": 0.15, "rtol": 0.15, "eps": EPS, "mode": "fwd"},
        {"f": f_add, "dtype": np.float32, "order": 1, "atol": 0.15, "rtol": 0.15, "eps": EPS, "mode": "rev"},
        {"f": f_add, "dtype": np.float32, "order": 2, "atol": 0.15, "rtol": 0.15, "eps": EPS, "mode": "fwd"},
        {"f": f_add, "dtype": np.float32, "order": 2, "atol": 0.15, "rtol": 0.15, "eps": EPS, "mode": "rev"},
        {"f": f_add, "dtype": np.float64, "order": 1, "atol": 1e-5, "rtol": 1e-5, "eps": EPS, "mode": "fwd"},
        {"f": f_add, "dtype": np.float64, "order": 1, "atol": 1e-5, "rtol": 1e-5, "eps": EPS, "mode": "rev"},
        {"f": f_add, "dtype": np.float64, "order": 2, "atol": 1e-5, "rtol": 1e-5, "eps": EPS, "mode": "fwd"},
        {"f": f_add, "dtype": np.float64, "order": 2, "atol": 1e-5, "rtol": 1e-5, "eps": EPS, "mode": "rev"},
        {"f": f_sub, "dtype": np.float32, "order": 1, "atol": 0.15, "rtol": 0.15, "eps": EPS, "mode": "fwd"},
        {"f": f_sub, "dtype": np.float32, "order": 1, "atol": 0.15, "rtol": 0.15, "eps": EPS, "mode": "rev"},
        {"f": f_sub, "dtype": np.float32, "order": 2, "atol": 0.15, "rtol": 0.15, "eps": EPS, "mode": "fwd"},
        {"f": f_sub, "dtype": np.float32, "order": 2, "atol": 0.15, "rtol": 0.15, "eps": EPS, "mode": "rev"},
        {"f": f_sub, "dtype": np.float64, "order": 1, "atol": 1e-5, "rtol": 1e-5, "eps": EPS, "mode": "fwd"},
        {"f": f_sub, "dtype": np.float64, "order": 1, "atol": 1e-5, "rtol": 1e-5, "eps": EPS, "mode": "rev"},
        {"f": f_sub, "dtype": np.float64, "order": 2, "atol": 1e-5, "rtol": 1e-5, "eps": EPS, "mode": "fwd"},
        {"f": f_sub, "dtype": np.float64, "order": 2, "atol": 1e-5, "rtol": 1e-5, "eps": EPS, "mode": "rev"},
        {"f": f_mul, "dtype": np.float32, "order": 1, "atol": 0.15, "rtol": 0.15, "eps": EPS, "mode": "fwd"},
        {"f": f_mul, "dtype": np.float32, "order": 1, "atol": 0.15, "rtol": 0.15, "eps": EPS, "mode": "rev"},
        {"f": f_mul, "dtype": np.float32, "order": 2, "atol": 0.15, "rtol": 0.15, "eps": EPS, "mode": "fwd"},
        {"f": f_mul, "dtype": np.float32, "order": 2, "atol": 0.15, "rtol": 0.15, "eps": EPS, "mode": "rev"},
        {"f": f_mul, "dtype": np.float64, "order": 1, "atol": 1e-5, "rtol": 1e-5, "eps": EPS, "mode": "fwd"},
        {"f": f_mul, "dtype": np.float64, "order": 1, "atol": 1e-5, "rtol": 1e-5, "eps": EPS, "mode": "rev"},
        {"f": f_mul, "dtype": np.float64, "order": 2, "atol": 1e-5, "rtol": 1e-5, "eps": EPS, "mode": "fwd"},
        {"f": f_mul, "dtype": np.float64, "order": 2, "atol": 1e-5, "rtol": 1e-5, "eps": EPS, "mode": "rev"},
        {"f": f_div, "dtype": np.float32, "order": 1, "atol": 0.15, "rtol": 0.15, "eps": EPS, "mode": "fwd"},
        {"f": f_div, "dtype": np.float32, "order": 1, "atol": 0.15, "rtol": 0.15, "eps": EPS, "mode": "rev"},
        {"f": f_div, "dtype": np.float64, "order": 1, "atol": 1e-5, "rtol": 1e-5, "eps": EPS, "mode": "fwd"},
        {"f": f_div, "dtype": np.float64, "order": 1, "atol": 1e-5, "rtol": 1e-5, "eps": EPS, "mode": "rev"}
    ]
    
    for config in param_combinations:
    
        f = config["f"]
        dtype = config["dtype"]

        if dtype == np.float32:
            atol = 0.15
            rtol = 0.15
        elif dtype == np.float64:
            atol = 1e-5
            rtol = 1e-5
        
        order = config["order"]
        eps = config["eps"]
        mode = config["mode"]
        
        tangent_diff_list, max_diff_list, portion_list = profile(f, dtype, atol, rtol, mode, order, eps)

        plt.figure()
        plt.hist(portion_list, bins=100)
        plt.ylabel("Frequency")
        plt.xlabel("Portion of jvp_diff/jax_threshold")
        plt.title(f"Portion_histogram_{f.__name__}_{dtype.__name__}_{atol}_{rtol}_{order}_{eps}_{mode}")
        plt.savefig(os.path.join(result_dir, f"portion_histogram_{f.__name__}_{dtype.__name__}_{atol}_{rtol}_{order}_{eps}_{mode}.png"))
        plt.close() 

        # create a new plot
        plt.figure()
        plt.hist(tangent_diff_list, bins=100)
        plt.ylabel("Frequency")
        plt.xlabel("jvp_diff")
        plt.title(f"jvp_diff_histogram_{f.__name__}_{dtype.__name__}_{atol}_{rtol}_{order}_{eps}_{mode}")
        plt.savefig(os.path.join(result_dir, f"jvp_diff_histogram_{f.__name__}_{dtype.__name__}_{atol}_{rtol}_{order}_{eps}_{mode}.png"))
        plt.close() 
    

if __name__ == "__main__":
    # main()
    primals = (np.float32(1.0), np.float32(2.0))
    vjp_AD, vjp_ND = check_gradsND(f_add, np.float32, primals, "rev", order=1, eps=1e-4, tangents=None)
    jvp_AD, jvp_ND = check_gradsND(f_add, np.float32, primals, "fwd", order=1, eps=1e-4, tangents=None)
    print("vjp_AD: ", vjp_AD)
    print("vjp_ND: ", vjp_ND)
    print("jvp_AD: ", jvp_AD)
    print("jvp_ND: ", jvp_ND)

    
    
        








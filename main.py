import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import os, sys
import itertools


proj_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_root)

from utils import utils, logger
from core import opSpec, rngFactory, exprTree, oracle

config = utils.load_yaml(os.path.join(proj_root, "configs.yaml"))


# Get the base filename without extension and core/ prefix
log_name = os.path.splitext(os.path.basename(__file__))[0]
logging = logger.setup_logger(log_name)
# Ensure the 'failures' directory exists
failures_root_dir = os.path.join(proj_root, 'failures')
results_dir = os.path.join(proj_root, 'results')
os.makedirs(failures_root_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

result_csv = os.path.join(results_dir, 'results.csv')

def run_tests(param_grid, num_tests):
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys,v)) for v in itertools.product(*values)]


    for params in param_combinations:
        expr_depth = params["expr_depth"]
        order = params["order"]
        mode = params["mode"]
        atol = params["atol"]
        eps = params["eps"]
        shape = params["shape"]
        dtype = params["dtype"]


        job_name = utils.param_to_jobname(params)
        num_failures = 0

        for i in range(num_tests):
        
            logging.debug(f"Shape: {shape}, expr_depth: {expr_depth}")
            tree, _, _ = exprTree.generate_expr_tree(expr_depth, shape, dtype)
            
            rng_factories = exprTree.collect_rng_factories(tree)
        
            inputs = [factory(shape=shape).astype(dtype) for factory in rng_factories]

            logging.debug(f"Inputs: {inputs}")
            logging.debug(f"Expr tree: {tree}")
            
            # Evaluate the expression with the generated inputs
            results = tree.evaluate(inputs)
            logging.debug(f"Results: {results}")

            # TODO: Figure out the atol, why will it change?
        
            status, err_msg = oracle.NDCheckDiagonal(lambda *args: tree.evaluate(list(args)), inputs, order=order, mode=mode, atol = atol,eps=eps)
            if status == "Fail":
                failures_dir = os.path.join(failures_root_dir, job_name)
                os.makedirs(failures_dir, exist_ok=True)
                num_failures += 1
                failure_hash = utils.generate_hash(tree, inputs)
                failure_file = os.path.join(failures_dir, f"failure_{failure_hash}.txt")
                with open(failure_file, "w") as f:
                    f.write(f"Error message: {err_msg}\n")
                    f.write("--------------------------------\n")
                    f.write(f"Expr tree: {tree}\n")
                    f.write("--------------------------------\n")
                    f.write(f"Inputs: {inputs}\n")
                    f.write("--------------------------------\n")
                    f.write(f"Atol: {atol}\n")
                    f.write(f"Step size: {eps}\n")
                    
                logging.error(f"{err_msg}")
            else:
                logging.info(f"Success")
        failure_rate = num_failures / num_tests
        result_file = os.path.join(results_dir, f"results_{job_name}.txt")
        with open(result_file, "w") as f:
            f.write(f"{failure_rate}")
    

def main():
    num_tests = 100

    # randomly generate some parameters
    # shape = (np.random.randint(1, 5), np.random.randint(1, 5))
    # expr_depth = np.random.randint(1, 5)

    # param_grid = {
    #     "expr_depth": [2, 3, 4],
    #     "order": [1, 2],
    #     "mode": ["fwd", "rev"],
    #     "atol": [1e-2, 1e-3, 1e-4, 1e-5],
    #     "eps": [1e-2, 1e-3, 1e-4, 1e-5],
    #     "shape": [(2,2), (3,3), (4,4)],
    #     "dtype": [np.float16, np.float32, np.float64]
    # }

    param_grid = {
        "expr_depth": [2],
        "order": [1],
        "mode": ["fwd", "rev"],
        "atol": [1e-2, 1e-3, 1e-4, 1e-5],
        "eps": [1e-2, 1e-3, 1e-4, 1e-5],
        "shape": [(1,2)],
        "dtype": [np.float16, np.float32, np.float64]
    }


    
    run_tests(param_grid, num_tests)
    
    utils.aggregate_results(param_grid, results_dir, result_csv)
  

          


if __name__ == "__main__":
    main()

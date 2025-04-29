import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import os, sys

proj_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_root)

from utils import utils, logger
from core import opSpec, rngFactory, exprTree, oracle

config = utils.load_yaml(os.path.join(proj_root, "configs.yaml"))


# Get the base filename without extension and core/ prefix
log_name = os.path.splitext(os.path.basename(__file__))[0]
logging = logger.setup_logger(log_name)


def main():

    # randomly generate some parameters
    # shape = (np.random.randint(1, 5), np.random.randint(1, 5))
    # depth = np.random.randint(1, 5)
    shape = (2, 2)
    depth = 3
    logging.debug(f"Shape: {shape}, Depth: {depth}")
    expr_tree, _ = exprTree.generate_expr_tree(depth, shape)
    
    rng_factories = exprTree.collect_rng_factories(expr_tree)
  
    inputs = [factory(shape=shape) for factory in rng_factories]
    logging.debug(f"Inputs: {inputs}")
    logging.debug(f"Expr tree: {expr_tree}")
    
    # Evaluate the expression with the generated inputs
    results = expr_tree.evaluate(inputs)
    logging.debug(f"Results: {results}")

    oracle.check_jvp(expr_tree.evaluate, inputs)


if __name__ == "__main__":
    main()
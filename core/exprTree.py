import numpy as np

import os, sys

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_root)

from utils import utils, logger
from core import opSpec, rngFactory

config = utils.load_yaml(os.path.join(proj_root, "configs.yaml"))

# Get the base filename without extension and core/ prefix
log_name = os.path.splitext(os.path.basename(__file__))[0]
logging = logger.setup_logger(log_name)

class Node:
    def evaluate(self):
        raise NotImplementedError("Subclasses must implement this method")
    def __str__(self):
        raise NotImplementedError("Subclasses must implement this method")

class Var(Node):
    def __init__(self, idx, value):
        self.idx = idx
        self.value = value
    def evaluate(self):
        return self.value
    def __str__(self):
        return f"x_{self.idx}"

class Const(Node):
    def __init__(self, value):
        self.value = value
    def evaluate(self):
        return self.value
    def __str__(self):
        return f"{self.value}"

class UnaryOp(Node):
    def __init__(self, operand, child):
        self.operand = operand
        self.child = child
    def evaluate(self):
        return self.operand(self.child.evaluate())
    def __str__(self):
        return f"{self.operand.__name__}({self.child})"

class BinaryOp(Node):
    def __init__(self, operand, left, right):
        self.operand = operand
        self.left = left
        self.right = right
    def evaluate(self):
        return self.operand(self.left.evaluate(), self.right.evaluate())
    def __str__(self):
        return f"{self.operand.__name__}({self.left}, {self.right})"
    
def generate_expr_tree(depth, shape=(1,1), dtype=float, rng_factory=None):
    var_idx = 0
    unary_specs = [s for s in opSpec.opSpec_list if s.nargs == 1]
    binary_specs = [s for s in opSpec.opSpec_list if s.nargs == 2]

    if depth == 0:
        # Choose a variable or a constant
        if np.random.rand() < 0.5:
            # Generate variable value using rng_factory if available, otherwise use uniform
            if rng_factory is not None:
                value = rng_factory(shape=shape)
            else:
                value = np.random.uniform(size=shape)
            var_idx += 1
            logging.debug(f"x_{var_idx}: {value}")
            return Var(var_idx, value)
        else:
            # Choose a constant
            if rng_factory is not None:
                value = rng_factory(shape=shape)
            else:
                value = np.random.uniform(size=shape)
            return Const(value)
    if np.random.rand() < 0.5 and len(unary_specs) > 0:
        # Choose a unary operator
        op_spec = np.random.choice(unary_specs)
        operand = op_spec.op
        return UnaryOp(operand, generate_expr_tree(depth-1, shape, dtype, op_spec.rng_factory))
    elif len(binary_specs) > 0:
        op_spec = np.random.choice(binary_specs)
        operand = op_spec.op
        return BinaryOp(operand, 
                       left=generate_expr_tree(depth-1, shape, dtype, op_spec.rng_factory),
                       right=generate_expr_tree(depth-1, shape, dtype, op_spec.rng_factory))
    else:
        raise ValueError("No operators available")

if __name__ == "__main__":
    expr_tree = generate_expr_tree(2)
    print(expr_tree)
    results  = expr_tree.evaluate()
    print(results)
    # print(expr_tree_to_jax{self.operand.__name__}_func(expr_tree))

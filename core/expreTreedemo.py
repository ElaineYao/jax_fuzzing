import jax
from oracle import check_grads_ND
from jax import lax

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
    def evaluate(self, inputs=None):
        raise NotImplementedError("Subclasses must implement this method")
    def __str__(self):
        raise NotImplementedError("Subclasses must implement this method")
    
class Var(Node):
    def __init__(self, index, dtype):
        self.index = index
        self.dtype = dtype

    def evaluate(self, inputs=None):
        return inputs[self.index]  # only one input for now

    def __str__(self):
        return f"x{self.index}"

class Const(Node):
    def __init__(self, value, dtype):
        self.value = value
        self.dtype = dtype

    def evaluate(self, inputs=None):
        return self.value

    def __str__(self):
        return str(self.value)

# TODO: How should I generate a function like this?
'''
def f(x1, x2):
    return x1+x2
'''
class BinaryOp(Node):
    def __init__(self, op_spec, left, right, dtype):
        self.op_spec = op_spec
        self.op = op_spec.op
        self.left = left
        self.right = right
        self.dtype = dtype
    
    def evaluate(self, inputs=None):
        return self.op(self.left.evaluate(inputs), self.right.evaluate(inputs))
    
    def __str__(self):
        return f"({self.left} {self.op_spec.name} {self.right})"

# Figure out how to control the distribution of the inputs
def generate_expr_tree(depth, dtype, n_args):
    unary_specs = [s for s in opSpec.opSpec_list if s.nargs == 1]
    binary_specs = [s for s in opSpec.opSpec_list if s.nargs == 2]

    if depth == 0:
        if np.random.rand() < 0.7:
            index = np.random.randint(n_args)
            return Var(index, dtype)
        else:
            value = dtype(np.random.uniform())
            return Const(value, dtype)
    else:
        op_specs = np.random.choice(binary_specs)
        left = generate_expr_tree(depth-1, dtype, n_args)
        right = generate_expr_tree(depth-1, dtype, n_args)
        return BinaryOp(op_specs, left, right, dtype)


def make_function_from_tree(expr_tree):
    def f(*args):
        return expr_tree.evaluate(args)
    return f



if __name__ == "__main__":
    
    
    dtype = np.float32
    n_args = 2
    expr_tree = generate_expr_tree(depth=3, dtype=dtype, n_args=n_args)
    print(expr_tree)
    f = make_function_from_tree(expr_tree)

    # tangents = (0.3, 1.5)
    cotangent = 2.0

    # primals = (1.0, 2.0)
    # def f(x):
    #     return x

    primals = (1.0, 2.0)
    tangents = (0.3, 1.5)
    result, result_ND= check_grads_ND(f, primals, cotangent, tangents, order=1, mode='fwd')
    print(result, result_ND)
    
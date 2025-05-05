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
    def __init__(self, idx, rng_factory, shape, dtype=float):
        self.idx = idx
        self.rng_factory = rng_factory
        self.shape = shape
        self.dtype = dtype
    def evaluate(self, inputs=None):
        if inputs is None:
            return self.rng_factory(shape=self.shape).astype(self.dtype)
        return inputs[self.idx].astype(self.dtype)
    def __str__(self):
        return f"x{self.idx}"

class Const(Node):
    def __init__(self, value, dtype=float):
        self.value = np.array(value, dtype=dtype)
    def evaluate(self, inputs=None):
        return self.value
    def __str__(self):
        if isinstance(self.value, np.ndarray):
            return f"jnp.array({np.array2string(self.value, separator=', ')})"
        return f"{self.value}"

class UnaryOp(Node):
    def __init__(self, operand, child, dtype=float):
        self.operand = operand
        self.child = child
        self.dtype = dtype
    def evaluate(self, inputs=None):
        return self.operand(self.child.evaluate(inputs).astype(self.dtype))
    def __str__(self):
        return f"lax.{self.operand.__name__}(\n    {self.child}\n)"

class BinaryOp(Node):
    def __init__(self, operand, left, right, dtype=float):
        self.operand = operand
        self.left = left
        self.right = right
        self.dtype = dtype
    def evaluate(self, inputs=None):
        return self.operand(self.left.evaluate(inputs).astype(self.dtype), self.right.evaluate(inputs).astype(self.dtype))
    def __str__(self):
        return f"lax.{self.operand.__name__}(\n    {self.left},\n    {self.right}\n)"
    
def generate_expr_tree(depth, shape=(1,1), dtype=float, rng_factory=None, var_idx=0, has_variable=False):
    unary_specs = [s for s in opSpec.opSpec_list if s.nargs == 1]
    binary_specs = [s for s in opSpec.opSpec_list if s.nargs == 2]

    if depth == 0:
        # Ensure at least one variable is included
        if not has_variable or np.random.rand() < 0.5:
            return Var(var_idx, rng_factory, shape, dtype), var_idx + 1, True
        else:
            # Choose a constant with the same shape
            if rng_factory is not None:
                value = rng_factory(shape=shape).astype(dtype)
            else:
                value = np.random.uniform(size=shape).astype(dtype)
            return Const(value, dtype), var_idx, has_variable

    if np.random.rand() < 0.5 and len(unary_specs) > 0:
        # Choose a unary operator
        op_spec = np.random.choice(unary_specs)
        operand = op_spec.op
        child, new_var_idx, has_variable = generate_expr_tree(depth-1, shape, dtype, op_spec.rng_factory, var_idx, has_variable)
        return UnaryOp(operand, child, dtype), new_var_idx, has_variable
    elif len(binary_specs) > 0:
        op_spec = np.random.choice(binary_specs)
        operand = op_spec.op
        left, new_var_idx, has_variable = generate_expr_tree(depth-1, shape, dtype, op_spec.rng_factory, var_idx, has_variable)
        right, final_var_idx, has_variable = generate_expr_tree(depth-1, shape, dtype, op_spec.rng_factory, new_var_idx, has_variable)
        return BinaryOp(operand, left, right, dtype), final_var_idx, has_variable
    else:
        raise ValueError("No operators available")

def collect_rng_factories(node, factories=None):
    if factories is None:
        factories = []
    # if isinstance(node, (Var, Const)):
    if isinstance(node, Var):
        factories.append(node.rng_factory)
    elif isinstance(node, UnaryOp):
        collect_rng_factories(node.child, factories)
    elif isinstance(node, BinaryOp):
        collect_rng_factories(node.left, factories)
        collect_rng_factories(node.right, factories)
    return factories

if __name__ == "__main__":
    shape = (1,2)
    # Generate the expression tree
    dtype = np.float64
    expr_tree, _, _ = generate_expr_tree(3, shape, dtype)
    
    # Get the RNG factories needed for inputs
    rng_factories = collect_rng_factories(expr_tree)
    # print("Number of inputs needed:", len(rng_factories))
    
    # Generate inputs using the RNG factories
    inputs = [factory(shape=shape) for factory in rng_factories]

    print("import jax.numpy as jnp")
    print("from jax import lax")
    for i, input in enumerate(inputs):
        print(f"x{i} =", f"jnp.array({np.array2string(input, separator=', ')})", "dtype:", input.dtype)
    
    print("results =", expr_tree)
    print("print(results)")
    
    # Evaluate the expression with the generated inputs
    results = expr_tree.evaluate(inputs)
    print("#Results:", results)
    # Print the data type of the result
    print("Result dtype:", results.dtype)

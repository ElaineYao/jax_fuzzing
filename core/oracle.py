import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import os, sys
import yaml
import hashlib
import time
import jax.random as random


proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_root)

from utils import utils, logger
from core import opSpec, rngFactory

config = utils.load_yaml(os.path.join(proj_root, "configs.yaml"))


# Get the base filename without extension and core/ prefix
log_name = os.path.splitext(os.path.basename(__file__))[0]
logging = logger.setup_logger(log_name)

def numerical_grad(fn, x, eps, order):
    if order == 1:
        delta = eps
        return (fn(x + delta) - fn(x - delta)) / (2 * eps)
    else:
        # Recursive central difference formula
        return (numerical_grad(fn, x + eps, eps, order - 1) -
                numerical_grad(fn, x - eps, eps, order - 1)) / (2 * eps)

def autodiff_grad(fn, order, mode):
    for _ in range(order):
        if mode == "fwd":
            fn = jax.jacfwd(fn)
        elif mode == "rev":
            fn = jax.jacrev(fn)
        else:
            raise ValueError("Mode must be 'fwd' or 'rev'")
    return fn

def numerical_jvp(f, primals, tangents, eps):
    delta = tangents*eps
    f_pos = f(*(primals+delta))
    f_neg = f(*(primals - delta))
    numerical_jvp_res = (f_pos - f_neg)/(2*eps)
    return numerical_jvp_res

# def check_jvp(f, args, atol=None, rtol=None, eps=1e-5):

    
    # tangent = jax.random.normal(rng_key, shape=args.shape, dtype = dtype)
    # v_out, t_out = f_jvp(args, tangent)
    # v_out_expected = f(*args)
    # t_out_expected = numerical_jvp(f, args, tangent, eps=eps)

    # is_v_out_close = jax.numpy.allclose(v_out, v_out_expected, rtol=rtol, atol = atol, equal_nan=True)
    # logging.debug(f"V_out: {v_out}. V_out_expected: {v_out_expected}.")
    # if not is_v_out_close:
    #     logging.debug(f"V_out and V_out_expected isn't close because |V_out - V_out_expected| > atol +rtol*|V_out_expected|")

    # is_t_out_close = jax.numpy.allclose(t_out, t_out_expected, rtol=rtol, atol = atol, equal_nan=True)
    # logging.debug(f"T_out: {t_out}. T_out_expected: {t_out_expected}.")
    # if not is_t_out_close:
    #     logging.debug(f"T_out and T_out_expected isn't close because |T_out - T_out_expected| > atol +rtol*|T_out_expected|")
           

# def NDCheck_grads(fn, args, order, mode, atol=1e-5, rtol=None, eps=1e-3):
#     if mode == "fwd" and order == 1:
#         check_jvp(fn, args, atol=atol, rtol=rtol, eps=eps)
   

        # _check_jvp(f, partial(api.jvp, f), args, err_msg=fwd_msg)
        # _check_jvp = partial(check_jvp, atol=atol, rtol=rtol, eps=eps)
        # check_jvp()

# def NDCheckDiagonal(fn, inputs, order, mode="fwd", atol=1e-5, eps=1e-3):
#     # We only care about the diagonal of the Jacobian
#     inputs = jnp.asarray(inputs)

#     # Compute AD derivative of given order
#     if mode == "fwd":
#         AD_result = jax.jacfwd(fn)(inputs)
#         ND_result = numerical_grad(fn, inputs, eps, order)
#     elif mode == "rev":
#         AD_result = jax.jacrev(fn)(inputs)
#         ND_result = numerical_grad(fn, inputs, eps, order)
    
#     squeeze_results = jnp.squeeze(AD_result)
#     AD_result = jnp.diagonal(squeeze_results, axis1=-2, axis2=-1)
#     ND_result = jnp.squeeze(ND_result)
   
#     err_msg = ""
#     if jnp.allclose(AD_result, ND_result, atol=atol):
#         status =  "Success"
#     else:
#         status = "Fail"
#         err_msg = f"Mismatch:\nAD: {AD_result}\nND: {ND_result}\natol: {atol}, eps: {eps}"
#     return status, err_msg

if __name__ == "__main__":
    f = lambda x: jnp.sin(x)
    # x = jnp.array(1.0)
    x = jnp.array([[1.0, 0.0]])
    seed = int(time.time())
    key = random.PRNGKey(seed)
    # print(NDCheckDiagonal(f, x, order=1, mode="fwd", atol=1e-5, eps=1e-4))  # Should check ∂²f/∂x²



# def NDCheck(fn, inputs, order, mode, atol, eps):
#     err_msg = ""
#     try:
#         from jax.test_util import check_grads
#         check_grads(fn, 
#                     inputs, 
#                     order,
#                     modes=(mode,),
#                     atol=atol,
#                     eps=eps)
#     except Exception as e:
#         status = "Fail"
#         err_msg = str(e)
#     else:
#         status = "Success"
#     return status, err_msg
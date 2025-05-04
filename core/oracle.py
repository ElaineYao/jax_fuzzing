import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import os, sys
import yaml
import hashlib

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_root)

from utils import utils, logger
from core import opSpec, rngFactory

config = utils.load_yaml(os.path.join(proj_root, "configs.yaml"))


# Get the base filename without extension and core/ prefix
log_name = os.path.splitext(os.path.basename(__file__))[0]
logging = logger.setup_logger(log_name)


def NDCheck(fn, inputs, order, mode, atol, eps):
    err_msg = ""
    try:
        from jax.test_util import check_grads
        check_grads(fn, 
                    inputs, 
                    order,
                    modes=(mode,),
                    atol=atol,
                    eps=eps)
    except Exception as e:
        status = "Fail"
        err_msg = str(e)
    else:
        status = "Success"
    return status, err_msg
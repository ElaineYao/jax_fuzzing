import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import os, sys

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_root)

from utils import utils, logger
from core import opSpec, rngFactory

config = utils.load_yaml(os.path.join(proj_root, "configs.yaml"))


# Get the base filename without extension and core/ prefix
log_name = os.path.splitext(os.path.basename(__file__))[0]
logging = logger.setup_logger(log_name)

# TODO: add atol
# Jacobian-vector product (JVP) vs numerical JVP
def check_jvp(f, args, atol = float(config["atol"]), eps = float(config["EPS"])):
    rng = np.random.RandomState()
    tangent = jax.tree.map(partial(utils.rand_like, rng), args)
    logging.debug(f"tangent: {tangent}")
    v_out, t_out = jax.jvp(f, args, tangent)
    v_out_expected = f(*args)
    t_out_expected = utils.numerical_jvp(f, args, tangent, eps)
    logging.debug(f"v_out: {v_out}, v_out_expected: {v_out_expected}")
    logging.debug(f"t_out: {t_out}, t_out_expected: {t_out_expected}")
    assert jnp.allclose(v_out, v_out_expected, atol=atol)
    assert jnp.allclose(t_out, t_out_expected, atol=atol)

# TODO: add check vjp

if __name__ == "__main__":
    f = lambda x: x[0] + x[1]
    args = (jnp.array([1.0, 2.0]),)
    atol = 1e-3
    eps = 1e-5
    check_jvp(f, args, atol, eps)
    # check_jvp(f, args)
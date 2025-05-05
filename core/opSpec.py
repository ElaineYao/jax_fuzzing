import jax
from jax import lax
import numpy as np

import os, sys

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_root)

from utils import utils, dtypes
from core import rngFactory as rnf
config = utils.load_yaml(os.path.join(proj_root, "configs.yaml"))

class OpSpec:
    def __init__(self, op, nargs, rng_factory, dtypes=float, tol=config["default_tol"]):
        self.op = op
        self.nargs = nargs
        self.rng_factory = rng_factory
        self.dtypes = dtypes
        self.tol = tol
        self.dtypes = dtypes


opSpec_list = [
    # OpSpec(
    #     op=lax.log,
    #     nargs=1,
    #     rng_factory=rnf.rand_positive),
    # OpSpec(
    #     op=lax.pow,
    #     nargs=2,
    #     rng_factory=rnf.rand_positive),
    # OpSpec(
    #     op = lax.div,
    #     nargs = 2,
    #     rng_factory = rnf.rand_positive), # TODO: Constraints on both left and right operands
    OpSpec(
        op = lax.add,
        nargs = 2,
        rng_factory = rnf.rand_positive,
        dtypes = dtypes.float_dtypes)
]



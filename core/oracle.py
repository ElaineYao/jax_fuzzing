import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten
import jax.tree_util as tree_util
import os, sys

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_root)

from utils import utils, logger
from core import opSpec, rngFactory



EPS=1e-4




def numerical_jvp_1(f, primals, tangents, eps=EPS):

    delta = tuple(t * eps for t in tangents)
    f_pos = f(*(p + d for p, d in zip(primals, delta)))
    f_neg = f(*(p - d for p, d in zip(primals, delta)))
    numerical_jvp_res = tree_util.tree_map(lambda a, b: (a-b)/(2*eps), f_pos, f_neg)
    
    return numerical_jvp_res

def numerical_jvp_2(f, primals, tangents, eps=EPS):
        
    primals, tangents = primals[0], tangents[0]
    delta = tuple(t * eps for t in tangents)

    f_pos = f(tuple(p + d for p, d in zip(primals, delta)))
    f_neg = f(tuple(p - d for p, d in zip(primals, delta)))

    return jax.tree_util.tree_map(lambda a, b: (a - b) / (2 * eps), f_pos, f_neg)


def check_grads_jvp_1(f, primals, tangents, eps):
    
    _, jvp_AD = jax.jvp(f, primals, tangents)
    jvp_ND = numerical_jvp_1(f, primals, tangents)
    return jvp_AD, jvp_ND

def check_grads_jvp_2(f, primals, tangents, eps, order = 2):
    
    if order == 1:
        _, jvp_AD = jax.jvp(f, primals, tangents)
        jvp_ND = numerical_jvp_2(f, primals, tangents, eps)
        return jvp_AD, jvp_ND

    def higher_order_jvp(*x):
        return jax.jvp(f, *x, tangents)[1]

    return check_grads_jvp_2(higher_order_jvp, (primals,), (tangents,), eps, order - 1)


def check_grads_vjp_1(f, primals, cotangent, tangent, eps):

    _, vjp_f = jax.vjp(f, *primals)
    vjp_AD = vjp_f(cotangent)
    print("vjp_AD:", vjp_AD)
    vjp_AD_ip = utils.inner_prod(vjp_AD, tangent)
    jvp_ND = numerical_jvp_1(f, primals, tangent, eps)
    print("jvp_ND:", jvp_ND)
    jvp_ND_ip = utils.inner_prod(cotangent, jvp_ND)
    return vjp_AD_ip, jvp_ND_ip



def get_grads_ND_2(f, primals, tangents, eps, order = 2,):
    
    if order == 1:
        jvp_ND = numerical_jvp_2(f, primals, tangents, eps)
        return jvp_ND

    def higher_order_jvp(*x):
        return jax.jvp(f, *x, tangents)[1]

    return check_grads_jvp_2(higher_order_jvp, (primals,), (tangents,), eps, order - 1)

def get_grads_vjpAD_2(f, primals, cotangent, tangent, order=2):
    
    if order == 1:
        _, vjp_f = jax.vjp(f, *primals)
        vjp_AD = vjp_f(cotangent)

        return vjp_AD

    def higher_order_vjp(*x):

        _, vjp_f = jax.vjp(f, *x)
        return vjp_f(cotangent)

    return get_grads_vjpAD_2(higher_order_vjp, primals, (cotangent, cotangent), tangent, order - 1)


def check_grads_vjp_2(f, primals, cotangent, tangents, eps):

    vjp_AD = get_grads_vjpAD_2(f, primals, cotangent, tangents)

    _, jvp_ND = get_grads_ND_2(f, primals, tangents, eps)

    tangents_sq = jax.tree_util.tree_map(lambda x: x * x, tangents)
    cotangent_sq = jax.tree_util.tree_map(lambda x: x * x, cotangent)
    vjp_AD_ip = utils.inner_prod(vjp_AD, tangents_sq)
    jvp_ND_ip = utils.inner_prod(cotangent_sq, jvp_ND)

    return vjp_AD_ip, jvp_ND_ip

def check_grads_ND(f, primals, cotangent, tangents, order, mode, eps = EPS):
    if mode == 'fwd' and order ==1:
        return check_grads_jvp_1(f, primals, tangents, eps)
    elif mode == 'fwd' and order == 2:
        return check_grads_jvp_2(f, primals, tangents, eps)
    elif mode == 'rev' and order == 1:
        return check_grads_vjp_1(f, primals, cotangent, tangents, eps)
    elif mode == 'rev' and order == 2:
        return check_grads_vjp_2(f, primals, cotangent, tangents, eps)

if __name__ == "__main__":
    def f_add(a, b):
        return a+b

    def f_custom(a, b):
        return a*a*a + b*b*b

    tangents = (0.3, 1.5)
    cotangent = 2.0

    primals = (1.0, 2.0)

    result, result_ND= check_grads_ND(f_custom, primals, cotangent, tangents, order=2, mode='rev')
    print("Result of check_grads:", result)
    print("Result of numerical JVP:", result_ND)


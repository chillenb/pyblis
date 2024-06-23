import ctypes

import numpy as np

from pyblis import core
from pyblis.core import gint_t, libblis


def gemm(alpha, a, b, beta, c, transa=False, transb=False, conja=False, conjb=False):
    objalpha = core.bli_createscalar(alpha)
    objbeta = core.bli_createscalar(beta)
    ao = core.bli_obj_create_from(a)
    bo = core.bli_obj_create_from(b)
    co = core.bli_obj_create_from(c)

    core.bli_obj_set_conjtrans(
        core.get_blis_trans_t(transa, conja), ao
    )
    core.bli_obj_set_conjtrans(
        core.get_blis_trans_t(transb, conjb), bo
    )
    libblis.bli_gemm(
        ctypes.byref(objalpha),
        ctypes.byref(ao),
        ctypes.byref(bo),
        ctypes.byref(objbeta),
        ctypes.byref(co),
    )

def hemm(alpha, a, b, beta, c, sidea="L", uploa="L", conja=False, transb=False, conjb=False):
    objalpha = core.bli_createscalar(alpha)
    objbeta = core.bli_createscalar(beta)
    ao = core.bli_obj_create_from(a)
    bo = core.bli_obj_create_from(b)
    co = core.bli_obj_create_from(c)

    if conja:
        core.bli_obj_set_conjtrans(
            core.get_blis_trans_t(False, conja),
            ao
        )
    
    core.bli_obj_set_conjtrans(
        core.get_blis_trans_t(transb, conjb),
        bo
    )

    core.bli_obj_set_uplo(
        core.get_blis_uplo_t(uploa), co
    )

    
    libblis.bli_hemm(
        ctypes.byref(objalpha),
        ctypes.byref(ao),
        ctypes.byref(bo),
        ctypes.byref(objbeta),
        ctypes.byref(co),
    )
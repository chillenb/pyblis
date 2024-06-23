import ctypes

import numpy as np

from pyblis import core
from pyblis.core import libblis, gint_t

def check_vecargs(*args):
    if not args:
        pass
    assert args[0].ndim == 1
    N = args[0].size
    c = args[0].dtype.type
    for a in args[1:]:
        assert a.ndim == 1, f"ndim is {a.ndim}, expected 1"
        assert a.size == N, f"size was {a.size}, expected {N}"
        assert a.dtype.type == c, f"dtype was {a.dtype.type}, expected {c}"


def bli_addv(x, y, conj=False):
    check_vecargs(x, y)
    xo = core.bli_obj_create_from(x)
    yo = core.bli_obj_create_from(y)
    if conj:
        libblis.bli_obj_set_conj(core.BLIS_CONJ_NO_TRANSPOSE, ctypes.byref(xo))
    libblis.bli_addv(ctypes.byref(xo), ctypes.byref(yo))


def bli_amaxv(x):
    xo = core.bli_obj_create_from(x)
    into = core.bli_createscalar(0, typechar='l')
    retval = gint_t()
    libblis.bli_obj_create_1x1_with_attached_buffer(
        core.BLIS_INT, ctypes.byref(retval), ctypes.byref(into)
    )
    libblis.bli_amaxv(ctypes.byref(xo), ctypes.byref(into))
    return core.bli_readscalar(into)


def bli_axpyv(alpha, x, y, conj=False):
    objalpha = core.bli_createscalar(alpha)
    check_vecargs(x, y)
    if np.iscomplex(alpha):
        assert np.iscomplexobj(y)
    xo = core.bli_obj_create_from(x)
    yo = core.bli_obj_create_from(y)
    if conj:
        libblis.bli_obj_set_conj(core.BLIS_CONJ_NO_TRANSPOSE, ctypes.byref(xo))
    libblis.bli_axpyv(ctypes.byref(objalpha), ctypes.byref(xo), ctypes.byref(yo))


def bli_copyv(x, y, conj=False):
    check_vecargs(x, y)
    xo = core.bli_obj_create_from(x)
    yo = core.bli_obj_create_from(y)
    if conj:
        libblis.bli_obj_set_conj(core.BLIS_CONJ_NO_TRANSPOSE, ctypes.byref(xo))
    libblis.bli_copyv(ctypes.byref(xo), ctypes.byref(yo))


def bli_dotv(x, y, conjx=False, conjy=False):
    check_vecargs(x, y)
    xo = core.bli_obj_create_from(x)
    yo = core.bli_obj_create_from(y)
    if conjx:
        libblis.bli_obj_set_conj(core.BLIS_CONJ_NO_TRANSPOSE, ctypes.byref(xo))
    if conjy:
        libblis.bli_obj_set_conj(core.BLIS_CONJ_NO_TRANSPOSE, ctypes.byref(yo))
    rho = core._obj_t()
    resdtype = np.result_type(x, y)
    libblis.bli_obj_create_1x1(core.dtypes_to_blis[resdtype.char], ctypes.byref(rho))
    libblis.bli_dotv(ctypes.byref(xo), ctypes.byref(yo), ctypes.byref(rho))
    retval = core.bli_readscalar(rho)
    core.bli_obj_free(rho)
    return retval

import ctypes

import numpy as np

from pyblis import core
from pyblis.core import gint_t, libblis


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


def addv(x, y, conj=False):
    check_vecargs(x, y)
    xo = core.bli_obj_create_from(x)
    yo = core.bli_obj_create_from(y)
    if conj:
        libblis.bli_obj_set_conj(core.BLIS_CONJUGATE, ctypes.byref(xo))
    libblis.bli_addv(ctypes.byref(xo), ctypes.byref(yo))


def amaxv(x):
    xo = core.bli_obj_create_from(x)
    into = core.bli_createscalar(0, typechar="l")
    retval = gint_t()
    libblis.bli_obj_create_1x1_with_attached_buffer(
        core.BLIS_INT, ctypes.byref(retval), ctypes.byref(into)
    )
    libblis.bli_amaxv(ctypes.byref(xo), ctypes.byref(into))
    return core.bli_readscalar(into)


def axpyv(alpha, x, y, conj=False):
    objalpha = core.bli_createscalar(alpha)
    check_vecargs(x, y)
    if np.iscomplex(alpha):
        assert np.iscomplexobj(y)
    xo = core.bli_obj_create_from(x)
    yo = core.bli_obj_create_from(y)
    if conj:
        libblis.bli_obj_set_conj(core.BLIS_CONJUGATE, ctypes.byref(xo))
    libblis.bli_axpyv(ctypes.byref(objalpha), ctypes.byref(xo), ctypes.byref(yo))


def axpbyv(alpha, x, beta, y, conjx=False, conjy=False):
    objalpha = core.bli_createscalar(alpha)
    objbeta = core.bli_createscalar(beta)
    check_vecargs(x, y)
    if np.iscomplex(alpha) or np.iscomplex(beta):
        assert np.iscomplexobj(y)
    xo = core.bli_obj_create_from(x)
    yo = core.bli_obj_create_from(y)
    if conjx:
        libblis.bli_obj_set_conj(core.BLIS_CONJUGATE, ctypes.byref(xo))
    if conjy:
        libblis.bli_obj_set_conj(core.BLIS_CONJUGATE, ctypes.byref(yo))
    libblis.bli_axpbyv(
        ctypes.byref(objalpha),
        ctypes.byref(xo),
        ctypes.byref(objbeta),
        ctypes.byref(yo),
    )


def copyv(x, y, conj=False):
    check_vecargs(x, y)
    xo = core.bli_obj_create_from(x)
    yo = core.bli_obj_create_from(y)
    if conj:
        libblis.bli_obj_set_conj(core.BLIS_CONJUGATE, ctypes.byref(xo))
    libblis.bli_copyv(ctypes.byref(xo), ctypes.byref(yo))


def dotv(x, y, conjx=False, conjy=False):
    check_vecargs(x, y)
    xo = core.bli_obj_create_from(x)
    yo = core.bli_obj_create_from(y)
    if conjx:
        libblis.bli_obj_set_conj(core.BLIS_CONJUGATE, ctypes.byref(xo))
    if conjy:
        libblis.bli_obj_set_conj(core.BLIS_CONJUGATE, ctypes.byref(yo))
    resdtype = np.result_type(x, y)
    rho = core.bli_createscalar(0, resdtype.char)
    libblis.bli_dotv(ctypes.byref(xo), ctypes.byref(yo), ctypes.byref(rho))
    return core.bli_readscalar(rho)


def dotxv(alpha, x, y, beta, conjx=False, conjy=False):
    objalpha = core.bli_createscalar(alpha)
    objbeta = core.bli_createscalar(beta)
    check_vecargs(x, y)
    if np.iscomplex(alpha) or np.iscomplex(beta):
        assert np.iscomplexobj(y)
    xo = core.bli_obj_create_from(x)
    yo = core.bli_obj_create_from(y)
    if conjx:
        libblis.bli_obj_set_conj(core.BLIS_CONJUGATE, ctypes.byref(xo))
    if conjy:
        libblis.bli_obj_set_conj(core.BLIS_CONJUGATE, ctypes.byref(yo))
    resdtype = np.result_type(x, y)
    rho = core.bli_createscalar(0, resdtype.char)
    libblis.bli_dotxv(
        ctypes.byref(objalpha),
        ctypes.byref(xo),
        ctypes.byref(yo),
        ctypes.byref(objbeta),
        ctypes.byref(rho),
    )
    return core.bli_readscalar(rho)


def invertv(x, conj=False):
    xo = core.bli_obj_create_from(x)
    check_vecargs(x)
    if conj:
        libblis.bli_obj_set_conj(core.BLIS_CONJUGATE, ctypes.byref(xo))
    libblis.bli_invertv(ctypes.byref(xo))


def invscalv(alpha, x):
    objalpha = core.bli_createscalar(alpha)
    check_vecargs(x)
    xo = core.bli_obj_create_from(x)
    libblis.bli_invscalv(ctypes.byref(objalpha), ctypes.byref(xo))


def scalv(alpha, x):
    objalpha = core.bli_createscalar(alpha)
    check_vecargs(x)
    xo = core.bli_obj_create_from(x)
    libblis.bli_scalv(ctypes.byref(objalpha), ctypes.byref(xo))


def scal2v(alpha, x, y, conj=False):
    objalpha = core.bli_createscalar(alpha)
    check_vecargs(x, y)
    xo = core.bli_obj_create_from(x)
    yo = core.bli_obj_create_from(y)
    if conj:
        libblis.bli_obj_set_conj(core.BLIS_CONJUGATE, ctypes.byref(xo))
    libblis.bli_scal2v(ctypes.byref(objalpha), ctypes.byref(xo), ctypes.byref(yo))


def setv(alpha, x):
    objalpha = core.bli_createscalar(alpha)
    check_vecargs(x)
    xo = core.bli_obj_create_from(x)
    libblis.bli_setv(ctypes.byref(objalpha), ctypes.byref(xo))


def setrv(alpha, x):
    objalpha = core.bli_createscalar(alpha)
    check_vecargs(x)
    xo = core.bli_obj_create_from(x)
    libblis.bli_setrv(ctypes.byref(objalpha), ctypes.byref(xo))


def setiv(alpha, x):
    objalpha = core.bli_createscalar(alpha)
    check_vecargs(x)
    xo = core.bli_obj_create_from(x)
    libblis.bli_setiv(ctypes.byref(objalpha), ctypes.byref(xo))


def subv(x, y, conj=False):
    check_vecargs(x, y)
    xo = core.bli_obj_create_from(x)
    yo = core.bli_obj_create_from(y)
    if conj:
        libblis.bli_obj_set_conj(core.BLIS_CONJUGATE, ctypes.byref(xo))
    libblis.bli_subv(ctypes.byref(xo), ctypes.byref(yo))


def swapv(x, y):
    check_vecargs(x, y)
    xo = core.bli_obj_create_from(x)
    yo = core.bli_obj_create_from(y)
    libblis.bli_swapv(ctypes.byref(xo), ctypes.byref(yo))

def axpy2v(alphax, alphay, x, y, z, conjx=False, conjy=False):
    objalphax = core.bli_createscalar(alphax)
    objalphay = core.bli_createscalar(alphay)
    check_vecargs(x, y, z)
    if np.iscomplex(alphax) or np.iscomplex(alphay):
        assert np.iscomplexobj(y)
    xo = core.bli_obj_create_from(x)
    yo = core.bli_obj_create_from(y)
    zo = core.bli_obj_create_from(z)
    if conjx:
        libblis.bli_obj_set_conj(core.BLIS_CONJUGATE, ctypes.byref(xo))
    if conjy:
        libblis.bli_obj_set_conj(core.BLIS_CONJUGATE, ctypes.byref(yo))
    libblis.bli_axpy2v(
        ctypes.byref(objalphax),
        ctypes.byref(objalphay),
        ctypes.byref(xo),
        ctypes.byref(yo),
        ctypes.byref(zo),
    )

def dotaxpyv(alpha, x, y, z, conjx=False, conjy=False):
    objalpha = core.bli_createscalar(alpha)
    check_vecargs(x, y, z)
    if np.iscomplex(alpha):
        assert np.iscomplexobj(y)
    xo = core.bli_obj_create_from(x)
    yo = core.bli_obj_create_from(y)
    zo = core.bli_obj_create_from(z)
    if conjx:
        libblis.bli_obj_set_conj(core.BLIS_CONJUGATE, ctypes.byref(xo))
    if conjy:
        libblis.bli_obj_set_conj(core.BLIS_CONJUGATE, ctypes.byref(yo))
    libblis.bli_dotaxpyv(
        ctypes.byref(objalpha),
        ctypes.byref(xo),
        ctypes.byref(yo),
        ctypes.byref(zo),
    )

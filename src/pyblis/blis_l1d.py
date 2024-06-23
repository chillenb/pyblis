import ctypes

from pyblis import core
from pyblis.core import libblis


def check_l1dargs(*args):
    if not args:
        pass
    assert args[0].ndim == 1
    N = args[0].size
    c = args[0].dtype.type
    for a in args[1:]:
        assert a.ndim == 1, f"ndim is {a.ndim}, expected 1"
        assert a.size == N, f"size was {a.size}, expected {N}"
        assert a.dtype.type == c, f"dtype was {a.dtype.type}, expected {c}"


def addd(a, b, diag_offset_a=0, unit_diag_a=False, transa=False, conja=False):
    check_l1dargs(a, b)
    ao = core.bli_obj_create_from(a)
    bo = core.bli_obj_create_from(b)
    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transa, conja), ao)
    core.bli_obj_set_diag_offset(diag_offset_a, ao)
    if unit_diag_a:
        core.bli_obj_set_diag(core.BLIS_UNIT_DIAG, ao)

    libblis.bli_addd(ctypes.byref(ao), ctypes.byref(bo))


def axpyd(alpha, a, b, diag_offset_a=0, unit_diag_a=False, transa=False, conja=False):
    check_l1dargs(a, b)
    objalpha = core.bli_createscalar(alpha)
    ao = core.bli_obj_create_from(a)
    bo = core.bli_obj_create_from(b)
    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transa, conja), ao)
    core.bli_obj_set_diag_offset(diag_offset_a, ao)
    if unit_diag_a:
        core.bli_obj_set_diag(core.BLIS_UNIT_DIAG, ao)

    libblis.bli_axpyd(ctypes.byref(objalpha), ctypes.byref(ao), ctypes.byref(bo))


def coypd(a, b, diag_offset_a=0, unit_diag_a=False, transa=False, conja=False):
    check_l1dargs(a, b)
    ao = core.bli_obj_create_from(a)
    bo = core.bli_obj_create_from(b)
    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transa, conja), ao)
    core.bli_obj_set_diag_offset(diag_offset_a, ao)
    if unit_diag_a:
        core.bli_obj_set_diag(core.BLIS_UNIT_DIAG, ao)

    libblis.bli_copyd(ctypes.byref(ao), ctypes.byref(bo))


def invertd(a, diag_offset_a=0):
    ao = core.bli_obj_create_from(a)
    core.bli_obj_set_diag_offset(diag_offset_a, ao)
    libblis.bli_invertd(ctypes.byref(ao))


def scald(alpha, a, diag_offset_a=0, unit_diag_a=False, transa=False, conja=False):
    check_l1dargs(a)
    objalpha = core.bli_createscalar(alpha)
    ao = core.bli_obj_create_from(a)
    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transa, conja), ao)
    core.bli_obj_set_diag_offset(diag_offset_a, ao)
    if unit_diag_a:
        core.bli_obj_set_diag(core.BLIS_UNIT_DIAG, ao)

    libblis.bli_scald(ctypes.byref(objalpha), ctypes.byref(ao))


def scal2d(alpha, a, b, diag_offset_a=0, unit_diag_a=False, transa=False, conja=False):
    check_l1dargs(a, b)
    objalpha = core.bli_createscalar(alpha)
    ao = core.bli_obj_create_from(a)
    bo = core.bli_obj_create_from(b)
    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transa, conja), ao)
    core.bli_obj_set_diag_offset(diag_offset_a, ao)
    if unit_diag_a:
        core.bli_obj_set_diag(core.BLIS_UNIT_DIAG, ao)

    libblis.bli_scal2d(ctypes.byref(objalpha), ctypes.byref(ao), ctypes.byref(bo))


def setd(alpha, a, diag_offset_a=0):
    objalpha = core.bli_createscalar(alpha)
    ao = core.bli_obj_create_from(a)
    core.bli_obj_set_diag_offset(diag_offset_a, ao)
    libblis.bli_setd(ctypes.byref(objalpha), ctypes.byref(ao))


def setrd(alpha, a, diag_offset_a=0):
    objalpha = core.bli_createscalar(alpha)
    ao = core.bli_obj_create_from(a)
    core.bli_obj_set_diag_offset(diag_offset_a, ao)
    libblis.bli_setrd(ctypes.byref(objalpha), ctypes.byref(ao))


def setid(alpha, a, diag_offset_a=0):
    objalpha = core.bli_createscalar(alpha)
    ao = core.bli_obj_create_from(a)
    core.bli_obj_set_diag_offset(diag_offset_a, ao)
    libblis.bli_setid(ctypes.byref(objalpha), ctypes.byref(ao))


def shiftd(alpha, a, diag_offset_a=0):
    objalpha = core.bli_createscalar(alpha)
    ao = core.bli_obj_create_from(a)
    core.bli_obj_set_diag_offset(diag_offset_a, ao)
    libblis.bli_shiftd(ctypes.byref(objalpha), ctypes.byref(ao))


def subd(a, b, diag_offset_a=0, unit_diag_a=False, transa=False, conja=False):
    check_l1dargs(a, b)
    ao = core.bli_obj_create_from(a)
    bo = core.bli_obj_create_from(b)
    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transa, conja), ao)
    core.bli_obj_set_diag_offset(diag_offset_a, ao)
    if unit_diag_a:
        core.bli_obj_set_diag(core.BLIS_UNIT_DIAG, ao)

    libblis.bli_subd(ctypes.byref(ao), ctypes.byref(bo))


def xpbyd(a, beta, b, diag_offset_a=0, unit_diag_a=False, transa=False, conja=False):
    check_l1dargs(a, b)
    objbeta = core.bli_createscalar(beta)
    ao = core.bli_obj_create_from(a)
    bo = core.bli_obj_create_from(b)
    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transa, conja), ao)
    core.bli_obj_set_diag_offset(diag_offset_a, ao)
    if unit_diag_a:
        core.bli_obj_set_diag(core.BLIS_UNIT_DIAG, ao)

    libblis.bli_xpbyd(ctypes.byref(ao), ctypes.byref(objbeta), ctypes.byref(bo))

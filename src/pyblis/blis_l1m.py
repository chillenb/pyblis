import ctypes

from pyblis import core
from pyblis.core import libblis


def check_l1dargs(*args):
    if not args:
        pass
    assert args[0].ndim == 2
    N = args[0].size
    c = args[0].dtype.type
    for a in args[1:]:
        assert a.ndim == 1, f"ndim is {a.ndim}, expected 1"
        assert a.size == N, f"size was {a.size}, expected {N}"
        assert a.dtype.type == c, f"dtype was {a.dtype.type}, expected {c}"


def copym(
    a, b, diag_offset_a=0, unit_diag_a=False, uplo_a="D", transa=False, conja=False
):
    check_l1dargs(a, b)
    ao = core.bli_obj_create_from(a)
    bo = core.bli_obj_create_from(b)
    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transa, conja), ao)
    core.bli_obj_set_diag_offset(diag_offset_a, ao)
    core.bli_obj_set_uplo(core.get_blis_uplo_t(uplo_a), ao)
    if unit_diag_a:
        core.bli_obj_set_diag(core.BLIS_UNIT_DIAG, ao)

    libblis.bli_copym(ctypes.byref(ao), ctypes.byref(bo))


def invscalm(alpha, a, diag_offset_a=0, unit_diag_a=False, uplo_a="D"):
    check_l1dargs(a)
    objalpha = core.bli_createscalar(alpha)
    ao = core.bli_obj_create_from(a)
    core.bli_obj_set_diag_offset(diag_offset_a, ao)
    core.bli_obj_set_uplo(core.get_blis_uplo_t(uplo_a), ao)
    if unit_diag_a:
        core.bli_obj_set_diag(core.BLIS_UNIT_DIAG, ao)

    libblis.bli_invscalm(ctypes.byref(objalpha), ctypes.byref(ao))


def scalm(alpha, a, diag_offset_a=0, unit_diag_a=False, uplo_a="D"):
    check_l1dargs(a)
    objalpha = core.bli_createscalar(alpha)
    ao = core.bli_obj_create_from(a)
    core.bli_obj_set_diag_offset(diag_offset_a, ao)
    core.bli_obj_set_uplo(core.get_blis_uplo_t(uplo_a), ao)
    if unit_diag_a:
        core.bli_obj_set_diag(core.BLIS_UNIT_DIAG, ao)

    libblis.bli_scalm(ctypes.byref(objalpha), ctypes.byref(ao))


def scal2m(
    alpha,
    a,
    b,
    diag_offset_a=0,
    unit_diag_a=False,
    uplo_a="D",
    transa=False,
    conja=False,
):
    check_l1dargs(a, b)
    objalpha = core.bli_createscalar(alpha)
    ao = core.bli_obj_create_from(a)
    bo = core.bli_obj_create_from(b)
    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transa, conja), ao)
    core.bli_obj_set_diag_offset(diag_offset_a, ao)
    core.bli_obj_set_uplo(core.get_blis_uplo_t(uplo_a), ao)
    if unit_diag_a:
        core.bli_obj_set_diag(core.BLIS_UNIT_DIAG, ao)

    libblis.bli_scal2m(ctypes.byref(objalpha), ctypes.byref(ao), ctypes.byref(bo))


def setm(alpha, a, diag_offset_a=0, unit_diag_a=False, uplo_a="D"):
    check_l1dargs(a)
    objalpha = core.bli_createscalar(alpha)
    ao = core.bli_obj_create_from(a)
    core.bli_obj_set_diag_offset(diag_offset_a, ao)
    core.bli_obj_set_uplo(core.get_blis_uplo_t(uplo_a), ao)
    if unit_diag_a:
        core.bli_obj_set_diag(core.BLIS_UNIT_DIAG, ao)

    libblis.bli_setm(ctypes.byref(objalpha), ctypes.byref(ao))


def setrm(alpha, a, diag_offset_a=0, unit_diag_a=False, uplo_a="D"):
    check_l1dargs(a)
    objalpha = core.bli_createscalar(alpha)
    ao = core.bli_obj_create_from(a)
    core.bli_obj_set_diag_offset(diag_offset_a, ao)
    core.bli_obj_set_uplo(core.get_blis_uplo_t(uplo_a), ao)
    if unit_diag_a:
        core.bli_obj_set_diag(core.BLIS_UNIT_DIAG, ao)

    libblis.bli_setrm(ctypes.byref(objalpha), ctypes.byref(ao))


def setim(alpha, a, diag_offset_a=0, unit_diag_a=False, uplo_a="D"):
    check_l1dargs(a)
    objalpha = core.bli_createscalar(alpha)
    ao = core.bli_obj_create_from(a)
    core.bli_obj_set_diag_offset(diag_offset_a, ao)
    core.bli_obj_set_uplo(core.get_blis_uplo_t(uplo_a), ao)
    if unit_diag_a:
        core.bli_obj_set_diag(core.BLIS_UNIT_DIAG, ao)

    libblis.bli_setim(ctypes.byref(objalpha), ctypes.byref(ao))


def subm(
    a, b, diag_offset_a=0, unit_diag_a=False, uplo_a="D", transa=False, conja=False
):
    check_l1dargs(a, b)
    ao = core.bli_obj_create_from(a)
    bo = core.bli_obj_create_from(b)
    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transa, conja), ao)
    core.bli_obj_set_diag_offset(diag_offset_a, ao)
    core.bli_obj_set_uplo(core.get_blis_uplo_t(uplo_a), ao)
    if unit_diag_a:
        core.bli_obj_set_diag(core.BLIS_UNIT_DIAG, ao)

    libblis.bli_subm(ctypes.byref(ao), ctypes.byref(bo))

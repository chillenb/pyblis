import ctypes

from pyblis import core
from pyblis.core import libblis


def gemm(alpha, a, b, beta, c, transa=False, transb=False, conja=False, conjb=False):
    objalpha = core.bli_createscalar(alpha)
    objbeta = core.bli_createscalar(beta)
    ao = core.bli_obj_create_from(a)
    bo = core.bli_obj_create_from(b)
    co = core.bli_obj_create_from(c)

    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transa, conja), ao)
    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transb, conjb), bo)
    libblis.bli_gemm(
        ctypes.byref(objalpha),
        ctypes.byref(ao),
        ctypes.byref(bo),
        ctypes.byref(objbeta),
        ctypes.byref(co),
    )


def gemmt(
    alpha,
    a,
    b,
    beta,
    c,
    transa=False,
    transb=False,
    conja=False,
    conjb=False,
    uplo_c="D",
):
    objalpha = core.bli_createscalar(alpha)
    objbeta = core.bli_createscalar(beta)
    ao = core.bli_obj_create_from(a)
    bo = core.bli_obj_create_from(b)
    co = core.bli_obj_create_from(c)

    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transa, conja), ao)
    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transb, conjb), bo)
    core.bli_obj_set_uplo(core.get_blis_uplo_t(uplo_c), co)

    libblis.bli_gemmt(
        ctypes.byref(objalpha),
        ctypes.byref(ao),
        ctypes.byref(bo),
        ctypes.byref(objbeta),
        ctypes.byref(co),
    )


def hemm(
    alpha, a, b, beta, c, side_a="L", uplo_a="D", conja=False, transb=False, conjb=False
):
    objalpha = core.bli_createscalar(alpha)
    objbeta = core.bli_createscalar(beta)
    ao = core.bli_obj_create_from(a)
    bo = core.bli_obj_create_from(b)
    co = core.bli_obj_create_from(c)

    if conja:
        core.bli_obj_set_conj(core.BLIS_CONJUGATE, ao)

    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transb, conjb), bo)

    core.bli_obj_set_uplo(core.get_blis_uplo_t(uplo_a), ao)

    libblis.bli_hemm(
        ctypes.c_int(core.get_blis_side_t(side_a)),
        ctypes.byref(objalpha),
        ctypes.byref(ao),
        ctypes.byref(bo),
        ctypes.byref(objbeta),
        ctypes.byref(co),
    )


def herk(alpha, a, beta, c, uplo_c="D", transa=False, conja=False):
    objalpha = core.bli_createscalar(alpha)
    objbeta = core.bli_createscalar(beta)
    ao = core.bli_obj_create_from(a)
    co = core.bli_obj_create_from(c)

    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transa, conja), ao)
    core.bli_obj_set_uplo(core.get_blis_uplo_t(uplo_c), co)

    libblis.bli_herk(
        ctypes.byref(objalpha),
        ctypes.byref(ao),
        ctypes.byref(objbeta),
        ctypes.byref(co),
    )


def her2k(
    alpha,
    a,
    b,
    beta,
    c,
    uplo_c="D",
    transa=False,
    transb=False,
    conja=False,
    conjb=False,
):
    objalpha = core.bli_createscalar(alpha)
    objbeta = core.bli_createscalar(beta)
    ao = core.bli_obj_create_from(a)
    bo = core.bli_obj_create_from(b)
    co = core.bli_obj_create_from(c)

    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transa, conja), ao)
    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transb, conjb), bo)
    core.bli_obj_set_uplo(core.get_blis_uplo_t(uplo_c), co)

    libblis.bli_her2k(
        ctypes.byref(objalpha),
        ctypes.byref(ao),
        ctypes.byref(bo),
        ctypes.byref(objbeta),
        ctypes.byref(co),
    )


def symm(
    alpha, a, b, beta, c, side_a="L", uplo_a="D", conja=False, transb=False, conjb=False
):
    objalpha = core.bli_createscalar(alpha)
    objbeta = core.bli_createscalar(beta)
    ao = core.bli_obj_create_from(a)
    bo = core.bli_obj_create_from(b)
    co = core.bli_obj_create_from(c)

    if conja:
        core.bli_obj_set_conj(core.BLIS_CONJUGATE, ao)

    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transb, conjb), bo)

    core.bli_obj_set_uplo(core.get_blis_uplo_t(uplo_a), ao)

    libblis.bli_symm(
        ctypes.c_int(core.get_blis_side_t(side_a)),
        ctypes.byref(objalpha),
        ctypes.byref(ao),
        ctypes.byref(bo),
        ctypes.byref(objbeta),
        ctypes.byref(co),
    )


def syrk(alpha, a, beta, c, uplo_c="D", transa=False, conja=False):
    objalpha = core.bli_createscalar(alpha)
    objbeta = core.bli_createscalar(beta)
    ao = core.bli_obj_create_from(a)
    co = core.bli_obj_create_from(c)

    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transa, conja), ao)
    core.bli_obj_set_uplo(core.get_blis_uplo_t(uplo_c), co)

    libblis.bli_syrk(
        ctypes.byref(objalpha),
        ctypes.byref(ao),
        ctypes.byref(objbeta),
        ctypes.byref(co),
    )


def syr2k(
    alpha,
    a,
    b,
    beta,
    c,
    uplo_c="D",
    transa=False,
    transb=False,
    conja=False,
    conjb=False,
):
    objalpha = core.bli_createscalar(alpha)
    objbeta = core.bli_createscalar(beta)
    ao = core.bli_obj_create_from(a)
    bo = core.bli_obj_create_from(b)
    co = core.bli_obj_create_from(c)

    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transa, conja), ao)
    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transb, conjb), bo)
    core.bli_obj_set_uplo(core.get_blis_uplo_t(uplo_c), co)

    libblis.bli_syr2k(
        ctypes.byref(objalpha),
        ctypes.byref(ao),
        ctypes.byref(bo),
        ctypes.byref(objbeta),
        ctypes.byref(co),
    )


def trmm(
    alpha, a, b, side_a="L", uplo_a="D", transa=False, conja=False, unit_diag_a=False
):
    objalpha = core.bli_createscalar(alpha)
    ao = core.bli_obj_create_from(a)
    bo = core.bli_obj_create_from(b)

    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transa, conja), ao)
    core.bli_obj_set_uplo(core.get_blis_uplo_t(uplo_a), ao)
    if unit_diag_a:
        core.bli_obj_set_diag(core.BLIS_UNIT_DIAG, ao)

    libblis.bli_trmm(
        ctypes.c_int(core.get_blis_side_t(side_a)),
        ctypes.byref(objalpha),
        ctypes.byref(ao),
        ctypes.byref(bo),
    )


def trmm3(
    alpha,
    a,
    b,
    beta,
    c,
    side_a="L",
    uplo_a="D",
    transa=False,
    conja=False,
    unit_diag_a=False,
    transb=False,
    conjb=False,
):
    objalpha = core.bli_createscalar(alpha)
    objbeta = core.bli_createscalar(beta)
    ao = core.bli_obj_create_from(a)
    bo = core.bli_obj_create_from(b)
    co = core.bli_obj_create_from(c)

    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transa, conja), ao)
    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transb, conjb), bo)
    core.bli_obj_set_uplo(core.get_blis_uplo_t(uplo_a), ao)
    if unit_diag_a:
        core.bli_obj_set_diag(core.BLIS_UNIT_DIAG, ao)

    libblis.bli_trmm3(
        ctypes.c_int(core.get_blis_side_t(side_a)),
        ctypes.byref(objalpha),
        ctypes.byref(ao),
        ctypes.byref(bo),
        ctypes.byref(objbeta),
        ctypes.byref(co),
    )


def trsm(
    alpha, a, b, side_a="L", uplo_a="D", transa=False, conja=False, unit_diag_a=False
):
    objalpha = core.bli_createscalar(alpha)
    ao = core.bli_obj_create_from(a)
    bo = core.bli_obj_create_from(b)

    core.bli_obj_set_conjtrans(core.get_blis_trans_t(transa, conja), ao)
    core.bli_obj_set_uplo(core.get_blis_uplo_t(uplo_a), ao)
    if unit_diag_a:
        core.bli_obj_set_diag(core.BLIS_UNIT_DIAG, ao)

    libblis.bli_trsm(
        ctypes.c_int(core.get_blis_side_t(side_a)),
        ctypes.byref(objalpha),
        ctypes.byref(ao),
        ctypes.byref(bo),
    )

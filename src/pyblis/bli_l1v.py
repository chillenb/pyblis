import ctypes

import numpy as np

from pyblis import defs, oapi
from pyblis.load_blis import _blis_lib as libblis
from pyblis.load_blis import config


def bli_addv(x, y, conj=False):
    xo = oapi.bli_obj_create_from(x)
    yo = oapi.bli_obj_create_from(y)
    if conj:
        libblis.bli_obj_set_conj(defs.BLIS_CONJ_NO_TRANSPOSE, ctypes.byref(xo))
    libblis.bli_addv(ctypes.byref(xo), ctypes.byref(yo))


def bli_amaxv(x):
    xo = oapi.bli_obj_create_from(x)
    into = defs._obj_t()
    libblis.bli_obj_create_1x1(defs.BLIS_INT, ctypes.byref(into))
    libblis.bli_amaxv(ctypes.byref(xo), ctypes.byref(into))
    intoptr = ctypes.addressof(into)
    adrbufptr = intoptr + config.obj_t_buffer_offset
    bufptr = ctypes.POINTER(config.gint_t).from_address(adrbufptr)
    retval = bufptr.contents.value
    oapi.bli_obj_free(into)
    return retval


def bli_axpyv(alpha, x, y, conj=False):
    objalpha = oapi.bli_createscalar(alpha)
    xo = oapi.bli_obj_create_from(x)
    yo = oapi.bli_obj_create_from(y)
    if conj:
        libblis.bli_obj_set_conj(defs.BLIS_CONJ_NO_TRANSPOSE, ctypes.byref(xo))
    libblis.bli_axpyv(ctypes.byref(objalpha), ctypes.byref(xo), ctypes.byref(yo))


def bli_copyv(x, y, conj=False):
    xo = oapi.bli_obj_create_from(x)
    yo = oapi.bli_obj_create_from(y)
    if conj:
        libblis.bli_obj_set_conj(defs.BLIS_CONJ_NO_TRANSPOSE, ctypes.byref(xo))
    libblis.bli_copyv(ctypes.byref(xo), ctypes.byref(yo))


def bli_dotv(x, y, conjx=False, conjy=False):
    xo = oapi.bli_obj_create_from(x)
    yo = oapi.bli_obj_create_from(y)
    if conjx:
        libblis.bli_obj_set_conj(defs.BLIS_CONJ_NO_TRANSPOSE, ctypes.byref(xo))
    if conjy:
        libblis.bli_obj_set_conj(defs.BLIS_CONJ_NO_TRANSPOSE, ctypes.byref(yo))
    rho = defs._obj_t()
    resdtype = np.result_type(x, y)
    libblis.bli_obj_create_1x1(defs.dtypes_to_blis[resdtype.char], ctypes.byref(rho))
    libblis.bli_dotv(ctypes.byref(xo), ctypes.byref(yo), ctypes.byref(rho))
    rhoptr = ctypes.addressof(rho)
    adrbufptr = rhoptr + config.obj_t_buffer_offset
    resarr = np.ctypeslib.as_array(
        ctypes.cast(
            ctypes.c_void_p.from_address(adrbufptr),
            ctypes.POINTER(ctypes.c_byte * resdtype.itemsize),
        ),
        shape=(resdtype.itemsize,),
    ).view(resdtype)
    retval = resarr[0]
    oapi.bli_obj_free(rho)
    return retval

import ctypes

import numpy as np

from pyblis import defs
from pyblis.load_blis import _blis_lib as libblis
from pyblis.load_blis import config


def bli_obj_create_from(mat):
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    assert mat.ndim == 2

    m = config.gint_t(mat.shape[0])
    n = config.gint_t(mat.shape[1])

    rs = config.gint_t(mat.strides[0] // mat.itemsize)
    cs = config.gint_t(mat.strides[1] // mat.itemsize)
    dt = defs.get_blis_dtype(mat)

    obj = defs._obj_t()
    libblis.bli_obj_create_with_attached_buffer(
        dt, m, n, mat.ctypes.data_as(ctypes.c_void_p), rs, cs, ctypes.byref(obj)
    )
    return obj


def bli_allocmatrix(shape, order="C", dtype=np.float64):
    dt = defs.dtypes_to_blis[np.dtype(dtype).char]
    assert len(shape) == 2
    m = config.gint_t(shape[0])
    n = config.gint_t(shape[1])
    if order == "C":
        rs = config.gint_t(1)
        cs = m
    elif order == "F":
        rs = n
        cs = config.gint_t(1)
    else:
        raise ValueError(f"Unknown contig: {order}")
    obj = defs._obj_t()
    libblis.bli_obj_create(dt, m, n, rs, cs, ctypes.byref(obj))
    adrobj = ctypes.addressof(obj)
    adrbufptr = adrobj + config.obj_t_buffer_offset
    bufptr = ctypes.c_void_p.from_address(adrbufptr)
    nbytes = np.prod(shape) * np.dtype(dtype).itemsize
    bufptr = ctypes.cast(bufptr, ctypes.POINTER(ctypes.c_byte))
    arr = np.ctypeslib.as_array(bufptr, shape=(nbytes,))
    return obj, arr.view(np.dtype(dtype)).reshape(shape, order=order)


def bli_createscalar(alpha):
    arr = np.array([alpha])
    dt = defs.get_blis_dtype(arr)
    obj = defs._obj_t()
    libblis.bli_obj_create_1x1_with_attached_buffer(
        dt, arr.ctypes.data_as(ctypes.c_void_p), ctypes.byref(obj)
    )
    return obj


def bli_obj_free(obj):
    libblis.bli_obj_free(ctypes.byref(obj))

import ctypes

import numpy as np

from pyblis.load_blis import config


def prep_1darray(arr):
    arr = np.asarray(arr)
    assert arr.ndim == 1
    inc = arr.strides[0] // arr.itemsize
    return arr.ctypes.data_as(ctypes.c_void_p), config.gint_t(inc)

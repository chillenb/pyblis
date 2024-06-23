import ctypes
import os
import sys
from ctypes import c_byte, c_int
from pathlib import Path

import numpy as np


def _load_blis():
    blis_search_paths = []
    if "LD_LIBRARY_PATH" in os.environ:
        blis_search_paths.extend(os.environ["LD_LIBRARY_PATH"].split(":"))
    blis_search_paths.extend(
        [
            Path(__file__).resolve(),
            Path(sys.prefix) / "lib",
            Path(sys.prefix) / "lib64",
        ]
    )
    for path in blis_search_paths:
        try:
            return np.ctypeslib.load_library("libblis", path)
        except OSError:
            pass
    return ctypes.CDLL("libblis.so")


_blis_lib = _load_blis()
_blis_lib.bli_init()

libblis = _blis_lib

if libblis.bli_info_get_int_type_size() == 64:
    int_type_size = 64
    gint_t = ctypes.c_int64
    obj_t_buffer_offset = 64
    guint_t = ctypes.c_uint64
elif libblis.bli_info_get_int_type_size() == 32:
    int_type_size = 32
    gint_t = ctypes.c_int32
    obj_t_buffer_offset = 40
    guint_t = ctypes.c_uint32
else:
    msg = "Unknown blis int type size"
    raise ValueError(msg)


# it's 160 bytes.
MAX_OBJ_T_SIZE = 256

BLIS_NO_TRANSPOSE = c_int(0)
BLIS_TRANSPOSE = c_int(8)
BLIS_CONJ_NO_TRANSPOSE = c_int(16)
BLIS_CONJ_TRANSPOSE = c_int(24)

BLIS_NO_CONJUGATE = c_int(0)
BLIS_CONJUGATE = c_int(16)

BLIS_ZEROS = c_int(0)
BLIS_LOWER = c_int(192)
BLIS_UPPER = c_int(96)
BLIS_DENSE = c_int(224)

BLIS_LEFT = c_int(0)
BLIS_RIGHT = c_int(1)

BLIS_NONUNIT_DIAG = c_int(0)
BLIS_UNIT_DIAG = c_int(256)

BLIS_NO_INVERT_DIAG = c_int(0)
BLIS_INVERT_DIAG = c_int(512)

BLIS_FLOAT = 0
BLIS_DOUBLE = 2
BLIS_SCOMPLEX = 1
BLIS_DCOMPLEX = 3
BLIS_INT = 4
BLIS_CONSTANT = 5
BLIS_DT_LO = 0
BLIS_DT_HI = 3

BLIS_REAL = c_int(0)
BLIS_COMPLEX = c_int(1)

BLIS_SINGLE_PREC = c_int(0)
BLIS_DOUBLE_PREC = c_int(2)

_obj_t = c_byte * MAX_OBJ_T_SIZE

blis_dt_to_typechar = {
    BLIS_FLOAT: "f",
    BLIS_DOUBLE: "d",
    BLIS_SCOMPLEX: "F",
    BLIS_DCOMPLEX: "D",
    BLIS_INT: ("l" if int_type_size == 64 else "i"),
}

typechar_to_blis_dt = {
    "f": BLIS_FLOAT,
    "d": BLIS_DOUBLE,
    "F": BLIS_SCOMPLEX,
    "D": BLIS_DCOMPLEX,
    "l": BLIS_INT,
    "i": BLIS_INT,
}


def get_blis_dtype(arr):
    return typechar_to_blis_dt[arr.dtype.char]


class _dcomplex(ctypes.Structure):
    _fields_ = [  # noqa: RUF012
        ("real", ctypes.c_double),
        ("imag", ctypes.c_double),
    ]


class _scomplex(ctypes.Structure):
    _fields_ = [  # noqa: RUF012
        ("real", ctypes.c_float),
        ("imag", ctypes.c_float),
    ]


typechar_to_ctypes = {
    "f": ctypes.c_float,
    "d": ctypes.c_double,
    "F": _scomplex,
    "D": _dcomplex,
    "l": ctypes.c_long,
    "i": ctypes.c_int,
}


class _obj_t(ctypes.Structure):
    _fields_ = [  # noqa: RUF012
        ("root", ctypes.c_void_p),
        ("off", gint_t * 2),
        ("dim", gint_t * 2),
        ("diag_off", gint_t),
        ("info", ctypes.c_uint32),
        ("info2", ctypes.c_uint32),
        ("elem_size", guint_t),
        ("buffer", ctypes.c_void_p),
        ("rs", gint_t),
        ("cs", gint_t),
        ("is", gint_t),
        ("scalar", _dcomplex),
        ("m_padded", gint_t),
        ("n_padded", gint_t),
        ("ps", gint_t),
        ("pd", gint_t),
        ("m_panel", gint_t),
        ("n_panel", gint_t),
        ("pad", ctypes.c_byte * 64),
    ]


# typedef struct obj_s
# {
# 	// Basic fields
# 	struct obj_s* root;

# 	dim_t         off[2];
# 	dim_t         dim[2];
# 	doff_t        diag_off;

# 	objbits_t     info;
# 	objbits_t     info2;
# 	siz_t         elem_size;

# 	void*         buffer;
# 	inc_t         rs;
# 	inc_t         cs;
# 	inc_t         is;

# 	// Bufferless scalar storage
# 	atom_t        scalar;

# 	// Pack-related fields
# 	dim_t         m_padded; // m dimension of matrix, including any padding
# 	dim_t         n_padded; // n dimension of matrix, including any padding
# 	inc_t         ps;       // panel stride (distance to next panel)
# 	inc_t         pd;       // panel dimension (the "width" of a panel:
# 	                        // usually MR or NR)
# 	dim_t         m_panel;  // m dimension of a "full" panel
# 	dim_t         n_panel;  // n dimension of a "full" panel

# } obj_t;

libblis.bli_amaxv.argtypes = [ctypes.POINTER(_obj_t), ctypes.POINTER(_obj_t)]
libblis.bli_amaxv.restype = None
libblis.bli_obj_create.argtypes = [
    c_int,
    gint_t,
    gint_t,
    gint_t,
    gint_t,
    ctypes.POINTER(_obj_t),
]
libblis.bli_obj_create.restype = None


def bli_obj_create_from(mat):
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    assert mat.ndim == 2

    m = gint_t(mat.shape[0])
    n = gint_t(mat.shape[1])

    rs = gint_t(mat.strides[0] // mat.itemsize)
    cs = gint_t(mat.strides[1] // mat.itemsize)
    dt = get_blis_dtype(mat)

    obj = _obj_t()
    libblis.bli_obj_create_with_attached_buffer(
        dt, m, n, mat.ctypes.data_as(ctypes.c_void_p), rs, cs, ctypes.pointer(obj)
    )
    return obj


def bli_allocmatrix(shape, order="C", dtype=np.float64):
    dt = ctypes.c_int(typechar_to_blis_dt[np.dtype(dtype).char])
    assert len(shape) == 2
    m = shape[0]
    n = shape[1]
    if order == "C":
        rs = 1
        cs = m
    elif order == "F":
        rs = n
        cs = 1
    else:
        msg = f"Unknown order: {order}"
        raise ValueError(msg)
    obj = _obj_t()

    libblis.bli_obj_create.argtypes = [
        ctypes.c_int,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.POINTER(_obj_t),
    ]
    libblis.bli_obj_create.restype = None

    libblis.bli_obj_create(dt, m, n, rs, cs, ctypes.byref(obj))

    nelem = (m - 1) * obj.rs + (n - 1) * obj.cs + 1
    nbytes = nelem * obj.elem_size
    bufptr = ctypes.cast(obj.buffer, ctypes.POINTER(ctypes.c_byte))
    arr = np.ctypeslib.as_array(bufptr, shape=(nbytes,))
    arr = arr.view(np.dtype(dtype))
    arr = np.lib.stride_tricks.as_strided(
        arr, shape, (obj.rs * arr.itemsize, obj.cs * arr.itemsize)
    )
    return obj, arr
    libblis.bli_obj_free(ctypes.byref(obj))
    return None, None


def bli_createscalar(alpha, typechar="d"):
    obj = _obj_t()
    libblis.bli_obj_scalar_init_detached(
        ctypes.c_int(typechar_to_blis_dt[typechar]), ctypes.byref(obj)
    )
    scalarptr = ctypes.cast(
        ctypes.pointer(obj.scalar), ctypes.POINTER(typechar_to_ctypes[typechar])
    )

    if typechar in ("F", "D"):
        alpha = complex(alpha)
        scalarptr.contents.value.real = alpha.real
        scalarptr.contents.value.imag = alpha.imag
    else:
        scalarptr.contents.value = alpha
    return obj


def bli_readscalar(obj):
    dt = obj.info & 0x7
    if dt == BLIS_FLOAT:
        return ctypes.cast(obj.buffer, ctypes.POINTER(ctypes.c_float)).contents.value
    if dt == BLIS_DOUBLE:
        return ctypes.cast(obj.buffer, ctypes.POINTER(ctypes.c_double)).contents.value
    if dt == BLIS_SCOMPLEX:
        scal = ctypes.cast(obj.buffer, ctypes.POINTER(_scomplex)).contents
        return scal.real + 1j * scal.imag
    if dt == BLIS_DCOMPLEX:
        scal = ctypes.cast(obj.buffer, ctypes.POINTER(_dcomplex)).contents
        return scal.real + 1j * scal.imag
    if dt == BLIS_INT:
        return ctypes.cast(obj.buffer, ctypes.POINTER(gint_t)).contents.value
    msg = "Unknown blis datatype"
    raise ValueError(msg)


def bli_obj_free(obj):
    libblis.bli_obj_free(ctypes.byref(obj))

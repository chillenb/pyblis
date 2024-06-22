from ctypes import c_byte, c_int

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

BLIS_FLOAT = c_int(0)
BLIS_DOUBLE = c_int(2)
BLIS_SCOMPLEX = c_int(1)
BLIS_DCOMPLEX = c_int(3)
BLIS_INT = c_int(4)
BLIS_CONSTANT = c_int(5)
BLIS_DT_LO = c_int(0)
BLIS_DT_HI = c_int(3)

BLIS_REAL = c_int(0)
BLIS_COMPLEX = c_int(1)

BLIS_SINGLE_PREC = c_int(0)
BLIS_DOUBLE_PREC = c_int(2)

_obj_t = c_byte * MAX_OBJ_T_SIZE

dtypes_to_blis = {
    "f": BLIS_FLOAT,
    "d": BLIS_DOUBLE,
    "F": BLIS_SCOMPLEX,
    "D": BLIS_DCOMPLEX,
}


def get_blis_dtype(arr):
    return dtypes_to_blis[arr.dtype.char]

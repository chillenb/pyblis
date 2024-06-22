import ctypes
import os
import sys
from dataclasses import dataclass
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


@dataclass
class BliConfig:
    int_type_size: int
    gint_t: object
    obj_t_buffer_offset: int


if _blis_lib.bli_info_get_int_type_size() == 64:
    config = BliConfig(int_type_size=64, gint_t=ctypes.c_int64, obj_t_buffer_offset=64)
elif _blis_lib.bli_info_get_int_type_size() == 32:
    config = BliConfig(int_type_size=32, gint_t=ctypes.c_int32, obj_t_buffer_offset=40)
else:
    raise ValueError("Unknown blis int type size")

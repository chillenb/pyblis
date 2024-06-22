import ctypes
import os
import sys
from pathlib import Path

import numpy as np


def load_blis():
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

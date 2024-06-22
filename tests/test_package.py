import importlib.metadata

import pyblis as m


def test_version():
    assert importlib.metadata.version("pyblis") == m.__version__

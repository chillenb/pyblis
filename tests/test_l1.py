import pytest
import numpy as np
from pyblis import blis_l1v

N = 100


@pytest.fixture(params=[np.float32, np.float64, np.complex64, np.complex128])
def setup_l1v(request):
    rng = np.random.default_rng(0)
    if np.iscomplex(request.param(1)):
        x = rng.random(N) + 1j * rng.random(N)
        y = rng.random(N) + 1j * rng.random(N)
        z = rng.random(N) + 1j * rng.random(N)
        alpha = rng.random() + 1j * rng.random()
        beta = rng.random() + 1j * rng.random()
    else:
        x = rng.random(N)
        y = rng.random(N)
        z = rng.random(N)
        alpha = rng.random()
        beta = rng.random()
    return x, y, z, alpha, beta


def test_addv(setup_l1v):
    x, y, z, alpha, beta = setup_l1v
    yc = y.copy()
    blis_l1v.addv(x, yc)
    assert np.allclose(x + y, yc)


def test_amaxv(setup_l1v):
    x, y, z, alpha, beta = setup_l1v
    assert blis_l1v.amaxv(x) == np.argmax(np.abs(x))


def test_axpyv(setup_l1v):
    x, y, z, alpha, beta = setup_l1v
    yc = y.copy()
    blis_l1v.axpyv(alpha, x, yc)
    assert np.allclose(alpha * x + y, yc)


def test_axpbyv(setup_l1v):
    x, y, z, alpha, beta = setup_l1v
    yc = y.copy()
    blis_l1v.axpbyv(alpha, x, beta, yc)
    assert np.allclose(alpha * x + beta * y, yc)


def test_copyv(setup_l1v):
    x, y, z, alpha, beta = setup_l1v
    yc = y.copy()
    blis_l1v.copyv(x, yc)
    assert np.allclose(x, yc)


def test_dotv(setup_l1v):
    x, y, z, alpha, beta = setup_l1v
    assert np.allclose(np.dot(x, y), blis_l1v.dotv(x, y))


def test_dotxv(setup_l1v):
    x, y, z, alpha, beta = setup_l1v
    assert np.allclose(np.dot(x, y), blis_l1v.dotxv(1.0, x, y, 0.0))


def test_invertv(setup_l1v):
    x, y, z, alpha, beta = setup_l1v
    xc = x.copy()
    blis_l1v.invertv(xc)
    assert np.allclose(1 / x, xc)


# def test_invscalv(setup_l1v):
#     x, y, z, alpha, beta = setup_l1v
#     xc = x.copy()
#     blis_l1v.invscalv(alpha, xc)
#     assert np.allclose(x / alpha, xc)


def test_scalv(setup_l1v):
    x, y, z, alpha, beta = setup_l1v
    xc = x.copy()
    blis_l1v.scalv(alpha, xc)
    assert np.allclose(alpha * x, xc)


def test_scal2v(setup_l1v):
    x, y, z, alpha, beta = setup_l1v
    yc = np.empty_like(y)
    blis_l1v.scal2v(alpha, x, yc)
    assert np.allclose(alpha * x, yc)


def test_setv(setup_l1v):
    x, y, z, alpha, beta = setup_l1v
    yc = np.empty_like(y)
    blis_l1v.setv(alpha, yc)
    assert np.allclose(alpha, yc)


def test_setrv(setup_l1v):
    x, y, z, alpha, beta = setup_l1v
    yc = y.copy()
    blis_l1v.setrv(alpha, yc)
    assert np.allclose(alpha.real, yc.real)


def test_setiv(setup_l1v):
    x, y, z, alpha, beta = setup_l1v
    yc = y.copy()
    blis_l1v.setiv(alpha, yc)
    assert np.allclose(alpha.imag, yc.imag)


def test_subv(setup_l1v):
    x, y, z, alpha, beta = setup_l1v
    yc = y.copy()
    blis_l1v.subv(x, yc)
    assert np.allclose(y - x, yc)

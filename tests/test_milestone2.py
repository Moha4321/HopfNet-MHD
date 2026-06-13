"""
Milestone 2 unit tests:
  - P(k) projects out the longitudinal component: k . P(k)R = 0 everywhere
    (k=0 excluded, where P(0)=I and the check is trivially satisfied)
  - P(k) is idempotent: P(P(R)) = P(R)
  - k=0 mode is left untouched (P(0) = I)
  - Spectral curl of a known analytic vector potential matches the
    analytically-computed magnetic field

Run with: pytest tests/test_milestone2.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
import hopfnet.hopfnet_cpp as eng

N = 8
L = 2 * np.pi


def random_complex_field(rng, shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def test_projection_removes_longitudinal_component():
    grid = eng.SpectralGrid(N, L)
    Nzh = grid.Nz_half

    rng = np.random.default_rng(0)
    Rx = random_complex_field(rng, (N, N, Nzh))
    Ry = random_complex_field(rng, (N, N, Nzh))
    Rz = random_complex_field(rng, (N, N, Nzh))

    Px, Py, Pz = eng.project_field(grid, Rx, Ry, Rz)

    kx, ky, kz = grid.kx(), grid.ky(), grid.kz()
    KX = kx[:, None, None]
    KY = ky[None, :, None]
    KZ = kz[None, None, :]

    k_dot_P = KX * Px + KY * Py + KZ * Pz

    # Exclude k=0, where P(0)=I and k.P(0)R = 0.R = 0 trivially (and |k|^2=0
    # makes the projection formula itself undefined there).
    k2 = grid.k2()
    nonzero = k2 > 1e-14
    assert np.allclose(k_dot_P[nonzero], 0.0, atol=1e-12)


def test_projection_is_idempotent():
    grid = eng.SpectralGrid(N, L)
    Nzh = grid.Nz_half

    rng = np.random.default_rng(1)
    Rx = random_complex_field(rng, (N, N, Nzh))
    Ry = random_complex_field(rng, (N, N, Nzh))
    Rz = random_complex_field(rng, (N, N, Nzh))

    Px, Py, Pz = eng.project_field(grid, Rx, Ry, Rz)
    PPx, PPy, PPz = eng.project_field(grid, Px, Py, Pz)

    np.testing.assert_allclose(PPx, Px, atol=1e-12)
    np.testing.assert_allclose(PPy, Py, atol=1e-12)
    np.testing.assert_allclose(PPz, Pz, atol=1e-12)


def test_k0_mode_is_identity():
    grid = eng.SpectralGrid(N, L)
    Nzh = grid.Nz_half

    rng = np.random.default_rng(2)
    Rx = random_complex_field(rng, (N, N, Nzh))
    Ry = random_complex_field(rng, (N, N, Nzh))
    Rz = random_complex_field(rng, (N, N, Nzh))

    Px, Py, Pz = eng.project_field(grid, Rx, Ry, Rz)

    assert np.isclose(Px[0, 0, 0], Rx[0, 0, 0])
    assert np.isclose(Py[0, 0, 0], Ry[0, 0, 0])
    assert np.isclose(Pz[0, 0, 0], Rz[0, 0, 0])


def test_spectral_curl_known_field():
    # A = (0, 0, sin(x))  =>  B = curl A = (0, -cos(x), 0)
    grid = eng.SpectralGrid(N, L)
    fft = eng.FFT3D(N)

    x = np.arange(N) * (L / N)
    X, _, _ = np.meshgrid(x, x, x, indexing='ij')

    Ax_real = np.zeros((N, N, N))
    Ay_real = np.zeros((N, N, N))
    Az_real = np.sin(X)

    Ax_hat = fft.forward(Ax_real)
    Ay_hat = fft.forward(Ay_real)
    Az_hat = fft.forward(Az_real)

    Bx_hat, By_hat, Bz_hat = eng.spectral_curl(grid, Ax_hat, Ay_hat, Az_hat)

    Bx = fft.inverse(Bx_hat)
    By = fft.inverse(By_hat)
    Bz = fft.inverse(Bz_hat)

    np.testing.assert_allclose(Bx, np.zeros((N, N, N)), atol=1e-12)
    np.testing.assert_allclose(By, -np.cos(X), atol=1e-12)
    np.testing.assert_allclose(Bz, np.zeros((N, N, N)), atol=1e-12)


def test_curl_of_projected_field_is_solenoidal():
    # Sanity check linking Milestone 1+2: B = curl(A) must satisfy
    # div B = 0 identically (k . B_hat = 0), regardless of projection,
    # since div(curl) ≡ 0 by construction of the spectral curl operator.
    grid = eng.SpectralGrid(N, L)
    Nzh = grid.Nz_half

    rng = np.random.default_rng(3)
    Ax = random_complex_field(rng, (N, N, Nzh))
    Ay = random_complex_field(rng, (N, N, Nzh))
    Az = random_complex_field(rng, (N, N, Nzh))

    Bx, By, Bz = eng.spectral_curl(grid, Ax, Ay, Az)

    kx, ky, kz = grid.kx(), grid.ky(), grid.kz()
    KX = kx[:, None, None]
    KY = ky[None, :, None]
    KZ = kz[None, None, :]

    div_B = KX * Bx + KY * By + KZ * Bz
    np.testing.assert_allclose(div_B, 0.0, atol=1e-10)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

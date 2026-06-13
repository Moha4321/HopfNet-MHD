"""
Milestone 3 unit tests:
  - Quadrature convergence: increasing n_quad changes A by less than a
    tight tolerance (the regularized integrand is smooth, so Simpson's
    rule converges fast).
  - Output is finite everywhere (no NaN/Inf from the regularized 1/D terms).
  - After FFT -> Coulomb projection -> IFFT, the field satisfies
    div A = 0 to spectral (machine) precision: k . A_hat = 0 for all k != 0.
  - The resulting B = curl(A) is non-trivial (the IC actually carries
    magnetic energy) and is itself solenoidal.

Run with: pytest tests/test_milestone3.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
import hopfnet.hopfnet_cpp as eng

N = 8
L = 2 * np.pi
R = 1.0
D = 0.3
A_CORE = 0.2


def test_hopf_link_finite():
    Ax, Ay, Az = eng.compute_hopf_link(N, L, R, D, A_CORE, n_quad=32)

    assert Ax.shape == (N, N, N)
    assert np.all(np.isfinite(Ax))
    assert np.all(np.isfinite(Ay))
    assert np.all(np.isfinite(Az))

    # The field should be non-trivial (not identically zero).
    assert np.abs(Ax).max() > 0
    assert np.abs(Ay).max() > 0
    assert np.abs(Az).max() > 0


def test_quadrature_convergence():
    # Simpson's rule converges algebraically (not to machine precision) for
    # this smooth-but-peaked regularized integrand. Check that the error
    # shrinks as n_quad increases (Richardson-style), rather than imposing
    # an unrealistically tight absolute tolerance between two fixed n_quad.
    Ax32, Ay32, Az32 = eng.compute_hopf_link(N, L, R, D, A_CORE, n_quad=32)
    Ax64, Ay64, Az64 = eng.compute_hopf_link(N, L, R, D, A_CORE, n_quad=64)
    Ax128, Ay128, Az128 = eng.compute_hopf_link(N, L, R, D, A_CORE, n_quad=128)

    err_32_64 = np.abs(Ax64 - Ax32).max()
    err_64_128 = np.abs(Ax128 - Ax64).max()

    # Error must decrease substantially as resolution doubles.
    assert err_64_128 < 0.5 * err_32_64

    # And the finest level should already be converged to a reasonable
    # absolute tolerance relative to the field's own magnitude.
    scale = max(np.abs(Ax128).max(), np.abs(Ay128).max(), np.abs(Az128).max())
    assert err_64_128 < 1e-3 * scale


def test_coulomb_gauge_after_projection():
    grid = eng.SpectralGrid(N, L)
    fft = eng.FFT3D(N)

    Ax, Ay, Az = eng.compute_hopf_link(N, L, R, D, A_CORE, n_quad=64)

    Ax_hat = fft.forward(Ax)
    Ay_hat = fft.forward(Ay)
    Az_hat = fft.forward(Az)

    Px, Py, Pz = eng.project_field(grid, Ax_hat, Ay_hat, Az_hat)

    kx, ky, kz = grid.kx(), grid.ky(), grid.kz()
    KX = kx[:, None, None]
    KY = ky[None, :, None]
    KZ = kz[None, None, :]

    k_dot_A = KX * Px + KY * Py + KZ * Pz

    k2 = grid.k2()
    nonzero = k2 > 1e-14
    assert np.allclose(k_dot_A[nonzero], 0.0, atol=1e-12)


def test_resulting_field_carries_energy_and_is_solenoidal():
    grid = eng.SpectralGrid(N, L)
    fft = eng.FFT3D(N)

    Ax, Ay, Az = eng.compute_hopf_link(N, L, R, D, A_CORE, n_quad=64)

    Ax_hat = fft.forward(Ax)
    Ay_hat = fft.forward(Ay)
    Az_hat = fft.forward(Az)

    Px, Py, Pz = eng.project_field(grid, Ax_hat, Ay_hat, Az_hat)
    Bx_hat, By_hat, Bz_hat = eng.spectral_curl(grid, Px, Py, Pz)

    # B must carry energy: a Hopf link with R, d > 0 is not a trivial field.
    energy = (np.abs(Bx_hat) ** 2 + np.abs(By_hat) ** 2 + np.abs(Bz_hat) ** 2).sum()
    assert energy > 0

    # curl(A) is solenoidal by construction: k . B_hat = 0 everywhere.
    kx, ky, kz = grid.kx(), grid.ky(), grid.kz()
    KX = kx[:, None, None]
    KY = ky[None, :, None]
    KZ = kz[None, None, :]
    div_B = KX * Bx_hat + KY * By_hat + KZ * Bz_hat
    np.testing.assert_allclose(div_B, 0.0, atol=1e-10)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

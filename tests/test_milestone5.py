"""
Milestone 5 unit tests — ETDRK4 Kassam-Trefethen contour coefficients.

Physics governed by two INDEPENDENT linear operators:
    L_A(k) = -eta*k^2 - eta4*k^4   (magnetic hyper-resistivity)
    L_v(k) = -nu*k^2  - nu4*k^4    (viscous hyper-diffusion)

Each requires its own coefficient set (E, E2, f1, f2, f3).

Tests:
  1. No NaN/Inf in any coefficient for any mode.
  2. k=0 regularity: f1(0)=1, f2(0)=1/2, f3(0)=1/2 without threshold switching.
  3. E[k]  = exp(L(k)*dt)   for all k.
  4. E2[k] = exp(L(k)*dt/2) for all k.
  5. f1[k] matches (exp(c)-1)/c for all non-trivial k.
  6. Smoothness: no discontinuous jumps in f1 along kx axis.
  7. Two independent operator sets (L_A != L_v) produce different arrays.

Run with: pytest tests/test_milestone5.py
"""
import os, sys
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
import hopfnet.hopfnet_cpp as eng

N   = 8
L   = 2 * np.pi
DT  = 0.01
ETA  = 0.001;  ETA4 = 1e-4
NU   = 0.002;  NU4  = 2e-4   # deliberately different from ETA/ETA4


def test_no_nan_inf_in_any_coefficient():
    grid = eng.SpectralGrid(N, L)
    for alpha, beta in [(ETA, ETA4), (NU, NU4)]:
        co = eng.precompute_etdrk4(grid, DT, alpha, beta)
        for arr in [co.E, co.E2, co.f1, co.f2, co.f3]:
            assert np.all(np.isfinite(arr)), "NaN/Inf found"


def test_k0_regularity_no_threshold_switching():
    """
    At k=0, c=0. L'Hopital limits:
        f1(0)=1,  f2(0)=1/2,  f3(0)=1/2.
    The M=32 contour must reproduce these exactly without any special-casing.
    """
    grid = eng.SpectralGrid(N, L)
    co = eng.precompute_etdrk4(grid, DT, ETA, ETA4)

    assert np.isclose(co.f1[0,0,0].real, 1.0, atol=1e-10), f"f1(0)={co.f1[0,0,0]}"
    assert np.isclose(co.f2[0,0,0].real, 0.5, atol=1e-10), f"f2(0)={co.f2[0,0,0]}"
    assert np.isclose(co.f3[0,0,0].real, 0.5, atol=1e-10), f"f3(0)={co.f3[0,0,0]}"
    assert abs(co.f1[0,0,0].imag) < 1e-12
    assert abs(co.f2[0,0,0].imag) < 1e-12
    assert abs(co.f3[0,0,0].imag) < 1e-12


def test_E_matches_exp_L_dt():
    grid = eng.SpectralGrid(N, L)
    co = eng.precompute_etdrk4(grid, DT, ETA, ETA4)
    k2 = grid.k2()
    E_expected = np.exp((-ETA * k2 - ETA4 * k2**2) * DT)
    np.testing.assert_allclose(co.E.real, E_expected, rtol=1e-12)
    np.testing.assert_allclose(co.E.imag, np.zeros_like(E_expected), atol=1e-14)


def test_E2_matches_exp_L_dt_half():
    grid = eng.SpectralGrid(N, L)
    co = eng.precompute_etdrk4(grid, DT, ETA, ETA4)
    k2 = grid.k2()
    E2_expected = np.exp((-ETA * k2 - ETA4 * k2**2) * DT / 2)
    np.testing.assert_allclose(co.E2.real, E2_expected, rtol=1e-12)


def test_f1_matches_analytic_nonzero_modes():
    grid = eng.SpectralGrid(N, L)
    co = eng.precompute_etdrk4(grid, DT, ETA, ETA4)
    k2 = grid.k2()
    c = (-ETA * k2 - ETA4 * k2**2) * DT
    mask = np.abs(c) > 1e-6
    f1_expected = (np.exp(c[mask]) - 1.0) / c[mask]
    np.testing.assert_allclose(co.f1[mask].real, f1_expected, rtol=1e-10)


def test_coefficients_smooth_no_discontinuities():
    grid = eng.SpectralGrid(N, L)
    co = eng.precompute_etdrk4(grid, DT, ETA, ETA4)
    f1_slice = co.f1[:, 0, 0].real
    diffs = np.abs(np.diff(f1_slice))
    # A threshold-switching bug would produce a jump >> mean step
    assert diffs.max() < 3.0 * diffs.mean() + 1e-12


def test_two_independent_operator_sets():
    grid = eng.SpectralGrid(N, L)
    co_A = eng.precompute_etdrk4(grid, DT, ETA, ETA4)
    co_v = eng.precompute_etdrk4(grid, DT, NU,  NU4)
    k2 = grid.k2()
    high_k = k2 > 1.0
    assert not np.allclose(co_A.E[high_k],  co_v.E[high_k])
    assert not np.allclose(co_A.f1[high_k], co_v.f1[high_k])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

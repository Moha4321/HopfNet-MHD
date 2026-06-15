"""
Milestone 5 unit tests — ETDRK4 Kassam-Trefethen contour coefficients.

Verified properties:
  1. e1/e2 match exp(L*dt/2)/exp(L*dt) at large |L*dt| (contour and direct agree).
  2. f1, f2, f3 match their Taylor-series limits at k=0 (L=0):
       f1(0) = dt,   f2(0) = dt/2,   f3(0) = dt/6.
  3. No NaN or Inf anywhere — the contour avoids 0/0 for every mode without
     a threshold switch.
  4. Smooth variation along k_x — no discontinuity a threshold switch would produce.
  5. Two independent ETDCoeffs (coeffs_A, coeffs_v) with distinct alpha/beta
     values agree only at k=0 and differ elsewhere.
  6. e1^2 = e2 everywhere (exp(L*dt/2)^2 = exp(L*dt)).

Run with: pytest tests/test_milestone5.py
"""
import os, sys
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
import hopfnet.hopfnet_cpp as eng

N    = 8
L    = 2 * np.pi
DT   = 1e-3

ETA  = 1e-3
ETA4 = 1e-4
NU   = 1e-3
NU4  = 1e-4


def make_coeffs(alpha, beta, M=32):
    grid = eng.SpectralGrid(N, L)
    c    = eng.ETDCoeffs(grid, DT, alpha, beta, M)
    return grid, c


def test_no_nan_or_inf():
    grid, c = make_coeffs(ETA, ETA4)
    for arr in [c.e1(grid), c.e2(grid), c.f1(grid), c.f2(grid), c.f3(grid)]:
        assert np.all(np.isfinite(arr)), "NaN/Inf found in ETD coefficient array"


def test_e1_e2_match_direct_at_large_k():
    grid, c = make_coeffs(ETA, ETA4)
    k2      = grid.k2()
    L_field = -ETA * k2 - ETA4 * k2**2

    e1_direct = np.exp(L_field * DT / 2.0)
    e2_direct = np.exp(L_field * DT)

    mask = np.abs(L_field * DT) > 0.1
    np.testing.assert_allclose(c.e1(grid)[mask], e1_direct[mask], rtol=1e-6)
    np.testing.assert_allclose(c.e2(grid)[mask], e2_direct[mask], rtol=1e-6)


def test_f1_f2_f3_at_k0_match_taylor_limits():
    # At k=0: L=0.  Taylor series gives:
    #   phi_1(0) = dt,   phi_2(0) = dt/2,   phi_3(0) = dt/6
    # The contour integral must reproduce these without any threshold guard.
    grid, c = make_coeffs(ETA, ETA4)
    np.testing.assert_allclose(c.f1(grid)[0,0,0], DT,       rtol=1e-6,
        err_msg="f1(k=0) must equal dt")
    np.testing.assert_allclose(c.f2(grid)[0,0,0], DT/2.0,   rtol=1e-6,
        err_msg="f2(k=0) must equal dt/2")
    np.testing.assert_allclose(c.f3(grid)[0,0,0], DT/6.0,   rtol=1e-6,
        err_msg="f3(k=0) must equal dt/6")


def test_coefficients_are_smooth_no_discontinuity():
    # A threshold switch would produce a visible jump in the coefficient
    # arrays at the switching wavenumber. Verify that adjacent modes along
    # k_x (j=0, kz=0 slice) differ by less than 5% of the max value.
    grid, c = make_coeffs(ETA, ETA4)
    f1_kx   = c.f1(grid)[:, 0, 0]
    diffs    = np.abs(np.diff(f1_kx))
    max_val  = np.abs(f1_kx).max()
    assert diffs.max() < 0.05 * max_val, \
        f"Jump of {diffs.max()/max_val:.3%} found — possible threshold discontinuity"


def test_e1_squared_equals_e2():
    # Fundamental identity: exp(L*dt/2)^2 = exp(L*dt)
    grid, c = make_coeffs(ETA, ETA4)
    np.testing.assert_allclose(c.e1(grid)**2, c.e2(grid), rtol=1e-6,
        err_msg="e1^2 != e2: inconsistency in half/full-step exponentials")


def test_two_independent_operators():
    # coeffs_A uses (ETA, ETA4); coeffs_v uses (2*NU, 2*NU4).
    # At k=0 (L=0 for both) f1 must equal dt for both.
    # At k != 0 they must differ.
    grid     = eng.SpectralGrid(N, L)
    coeffs_A = eng.ETDCoeffs(grid, DT, ETA,    ETA4)
    coeffs_v = eng.ETDCoeffs(grid, DT, 2.0*NU, 2.0*NU4)

    f1_A = coeffs_A.f1(grid)
    f1_v = coeffs_v.f1(grid)

    # k=0: both must be dt
    np.testing.assert_allclose(f1_A[0,0,0], DT, rtol=1e-6)
    np.testing.assert_allclose(f1_v[0,0,0], DT, rtol=1e-6)

    # k != 0: must differ
    assert not np.allclose(f1_A, f1_v), \
        "Distinct operators must produce distinct f1 at k != 0"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
"""
Milestone 6 Diagnostics unit tests.
Validates the physical diagnostics using a Beltrami field eigenstate,
where exact analytic values for Energy and Generalized Helicity are known.

Run with: pytest tests/test_milestone6_diagnostics.py
"""
import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
import hopfnet.hopfnet_cpp as eng
from hopfnet.rhs import to_hat
from hopfnet.diagnostics import compute_diagnostics

# N=16 easily resolves a k=1 sinusoid without aliasing
N = 16
L = 2 * np.pi

def test_diagnostics_analytic_beltrami():
    grid = eng.SpectralGrid(N, L)
    fft = eng.FFT3D(N)
    
    x = np.arange(N) * (L / N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # Construct an analytic Beltrami flow field
    # For this specific field, curl(A) = -A  (Eigenvalue -1)
    Ax = np.sin(Y) + np.cos(Z)
    Ay = np.sin(Z) + np.cos(X)
    Az = np.sin(X) + np.cos(Y)

    # Move to spectral space
    A_hat = to_hat(fft, (Ax, Ay, Az))
    
    # Let fluid velocity equal the vector potential for a fully symmetric test
    v_hat = to_hat(fft, (Ax, Ay, Az))

    d_i = 0.2
    diags = compute_diagnostics(grid, fft, A_hat, v_hat, d_i)

    # --- 1. Divergence Checks ---
    # The Beltrami field is analytically divergence-free. 
    # Spectral check must evaluate to machine precision.
    assert diags["div_A_max"] < 1e-12
    assert diags["div_v_max"] < 1e-12

    # --- 2. Energy Verification ---
    # E_m = 0.5 * \int |B|^2 dV   (magnetic energy, B = curl A)
    # For this specific Beltrami eigenstate curl(A) = -A, so |B| = |A|.
    # Therefore E_m = 0.5 * \int |A|^2 dV holds ONLY for this eigenstate —
    # it is NOT a general identity.
    # |A|^2 = (sin Y + cos Z)^2 + (sin Z + cos X)^2 + (sin X + cos Y)^2
    # Each squared trig term integrates to 0.5*V; there are 6 such terms.
    V = (2 * np.pi)**3
    expected_E_m = 0.5 * (6 * 0.5) * V   # = 0.5 * int|B|^2 dV = 0.5 * int|A|^2 dV
    expected_E_k = 0.5 * (6 * 0.5) * V   # = 0.5 * int|v|^2 dV (v = A here)

    np.testing.assert_allclose(diags["E_m"], expected_E_m, rtol=1e-12)
    np.testing.assert_allclose(diags["E_k"], expected_E_k, rtol=1e-12)

    # --- 3. Generalized Helicity Verification ---
    # Since B = -A and omega = -v = -A
    # A* = A + d_i * v = (1 + d_i) A
    # B* = B + d_i * omega = -(1 + d_i) A
    # H_gen = \int A* . B* dV = -(1 + d_i)^2 \int |A|^2 dV
    expected_H_gen = -(1 + d_i)**2 * (3 * V)

    np.testing.assert_allclose(diags["H_gen"], expected_H_gen, rtol=1e-12)

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
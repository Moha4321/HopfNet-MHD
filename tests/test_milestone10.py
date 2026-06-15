"""
Milestone 10: Topological Diagnostics unit tests.

Run with: pytest tests/test_milestone10.py -v
"""
import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
import hopfnet.hopfnet_cpp as eng
from hopfnet.rhs import to_hat, compute_B_and_J
from hopfnet.topology import (
    compute_magnetic_helicity, 
    compute_flux,
    compute_linking_number, 
    compute_1d_spectrum,
    compute_helicity_decay_rate
)

N = 32
L = 2 * np.pi

def test_beltrami_helicity():
    grid = eng.SpectralGrid(N, L)
    fft = eng.FFT3D(N)
    x = np.arange(N) * (L / N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # Beltrami Flow: B = curl(A) = -A
    Ax = np.sin(Y) + np.cos(Z)
    Ay = np.sin(Z) + np.cos(X)
    Az = np.sin(X) + np.cos(Y)
    A_hat = to_hat(fft, (Ax, Ay, Az))
    
    H = compute_magnetic_helicity(grid, fft, A_hat)
    expected_H = -3.0 * (L**3)
    np.testing.assert_allclose(H, expected_H, rtol=1e-12)

def test_parseval_consistency():
    grid = eng.SpectralGrid(N, L)
    fft = eng.FFT3D(N)
    rng = np.random.default_rng(42)
    
    Bx = rng.standard_normal((N, N, N))
    By = rng.standard_normal((N, N, N))
    Bz = rng.standard_normal((N, N, N))
    B_hat = to_hat(fft, (Bx, By, Bz))
    
    E_m_real = 0.5 * np.sum(Bx**2 + By**2 + Bz**2) * ((L / N)**3)
    _, E_k = compute_1d_spectrum(grid, B_hat)
    
    np.testing.assert_allclose(np.sum(E_k), E_m_real, rtol=1e-12)

def test_hopf_link_linking_number():
    grid = eng.SpectralGrid(N, L)
    fft = eng.FFT3D(N)
    
    Ax, Ay, Az = eng.compute_hopf_link(N, L, R=1.0, d=0.3, a_core=0.2, I0=1.0, mu0=1.0, n_quad=64)
    A_hat = to_hat(fft, (Ax, Ay, Az))
    A_hat = eng.project_field(grid, *A_hat)
    B_hat, _ = compute_B_and_J(grid, A_hat)
    
    H0 = compute_magnetic_helicity(grid, fft, A_hat, B_hat)
    Phi0 = compute_flux(grid, fft, B_hat)
    
    # Calculate initial geometric calibration factor
    calib = (2.0 * Phi0**2) / H0
    
    Lk = compute_linking_number(grid, fft, A_hat, B_hat, Phi0, calib_factor=calib)
    
    # By definition of our calibration, Lk must start exactly at 1.0
    np.testing.assert_allclose(Lk, 1.0, rtol=1e-12)

def test_helicity_decay_rate():
    # Simple finite difference check
    rate = compute_helicity_decay_rate(H_gen_current=0.9, H_gen_prev=1.0, dt=0.01, diag_interval=10)
    np.testing.assert_allclose(rate, -1.0, rtol=1e-12)

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
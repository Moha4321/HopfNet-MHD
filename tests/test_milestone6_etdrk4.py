"""
Milestone 6: ETDRK4 Integration unit tests.

Run with: pytest tests/test_milestone6_etdrk4.py
"""
import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
import hopfnet.hopfnet_cpp as eng
from hopfnet.etdrk4 import ETDRK4Integrator

N = 8
L = 2 * np.pi

def test_etdrk4_half_step_identity():
    grid = eng.SpectralGrid(N, L)
    fft = eng.FFT3D(N)
    dt = 0.01
    
    integrator = ETDRK4Integrator(grid, fft, dt, d_i=0.1, eta=0.01, eta4=1e-4, nu=0.01, nu4=1e-4)
    
    # The exact mathematical limit of Q = (e^(c/2) - 1)/c as c->0 is 1/2.
    # Check the k=0 mode (index 0,0,0) where c=0.
    q_limit = integrator.Q_A[0, 0, 0]
    
    assert np.isclose(q_limit.real, 0.5, atol=1e-12)
    assert np.isclose(q_limit.imag, 0.0, atol=1e-12)

def test_etdrk4_linear_exactness():
    grid = eng.SpectralGrid(N, L)
    fft = eng.FFT3D(N)
    dt = 0.1
    
    integrator = ETDRK4Integrator(grid, fft, dt, d_i=0.1, eta=0.1, eta4=1e-3, nu=0.1, nu4=1e-3)
    
    # Override the RHS evaluator to return pure zeros (linear decay only)
    def mock_rhs(A, v):
        z = np.zeros_like(A[0])
        return (z, z, z), (z, z, z)
    integrator._eval_rhs = mock_rhs
    
    rng = np.random.default_rng(42)
    def rand_field():
        return tuple(
            rng.standard_normal((N, N, grid.Nz_half)) + 1j * rng.standard_normal((N, N, grid.Nz_half)) 
            for _ in range(3)
        )
        
    A_n = eng.project_field(grid, *rand_field())
    v_n = eng.project_field(grid, *rand_field())
    
    # Take a step
    A_next, v_next = integrator.step(A_n, v_n)
    
    # In a purely linear system (N=0), ETD is mathematically EXACT: A_{n+1} = exp(L*dt) * A_n
    expected_A = tuple(integrator.c_A.E * A_n[i] for i in range(3))
    expected_v = tuple(integrator.c_v.E * v_n[i] for i in range(3))
    
    # Apply final projection to match the integrator's pipeline
    expected_A = eng.project_field(grid, *expected_A)
    expected_v = eng.project_field(grid, *expected_v)
    
    for i in range(3):
        np.testing.assert_allclose(A_next[i], expected_A[i], atol=1e-12)
        np.testing.assert_allclose(v_next[i], expected_v[i], atol=1e-12)

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
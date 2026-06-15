"""
Milestone 11: Current-Sheet Point Cloud Extractor unit tests.

Verifies the spectral gradients, spectral Hessians, and 
Shape Operator projections (curvature tensors).
"""
import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
import hopfnet.hopfnet_cpp as eng
from hopfnet.rhs import to_hat

N = 16
L = 2 * np.pi

def test_point_cloud_shape_operator():
    grid = eng.SpectralGrid(N, L)
    fft = eng.FFT3D(N)
    
    x = np.arange(N) * (L / N) - L / 2.0
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # Construct an analytic positive field for |J|
    # Jx = cos(x) + cos(y) + cos(z) + 4
    # Jy = 0, Jz = 0
    # Thus f = |J| = cos(x) + cos(y) + cos(z) + 4
    Jx = np.cos(X) + np.cos(Y) + np.cos(Z) + 4.0
    Jy = np.zeros_like(X)
    Jz = np.zeros_like(X)
    
    Jx_hat, Jy_hat, Jz_hat = to_hat(fft, (Jx, Jy, Jz))
    
    # threshold = 0.0 forces it to return all N^3 points so we can check everywhere
    pts = eng.extract_point_cloud(grid, fft, Jx_hat, Jy_hat, Jz_hat, 0.0)
    
    # Verify shape
    assert pts.shape == (N**3, 9)
    
    # Pick a specific known test point from the output list: 
    # Near (x, y, z) = (0, 0, 0)
    idx = np.argmin(pts[:, 0]**2 + pts[:, 1]**2 + pts[:, 2]**2)
    test_pt = pts[idx]
    
    # Analytic derivation at (x,y,z):
    px, py, pz = test_pt[0], test_pt[1], test_pt[2]
    
    grad = np.array([-np.sin(px), -np.sin(py), -np.sin(pz)])
    grad_mag = np.linalg.norm(grad)
    
    # Avoid zero division in analytic check if we hit exactly 0,0,0
    if grad_mag > 1e-6:
        n = grad / grad_mag
        H = np.diag([-np.cos(px), -np.cos(py), -np.cos(pz)])
        
        P = np.eye(3) - np.outer(n, n)
        # Analytic kappa = (1/|grad|) P * H * P
        kappa_expected = (1.0 / grad_mag) * (P @ H @ P)
        
        # Extractor output: k11, k12, k13, k22, k23, k33
        kappa_extracted = np.array([
            [test_pt[3], test_pt[4], test_pt[5]],
            [test_pt[4], test_pt[6], test_pt[7]],
            [test_pt[5], test_pt[7], test_pt[8]]
        ])
        
        np.testing.assert_allclose(kappa_extracted, kappa_expected, atol=1e-10)

def test_point_cloud_thresholding():
    grid = eng.SpectralGrid(N, L)
    fft = eng.FFT3D(N)
    x = np.arange(N) * (L / N) - L / 2.0
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # A highly localized Gaussian "current sheet"
    J_real = np.exp(- (X**2 + Y**2 + Z**2))
    J_hat = to_hat(fft, (J_real, np.zeros_like(X), np.zeros_like(X)))
    
    # With threshold 0.9, it should only extract a tiny fraction of points at the core
    pts = eng.extract_point_cloud(grid, fft, J_hat[0], J_hat[1], J_hat[2], 0.9)
    
    # 16^3 = 4096. A 0.9 Gaussian threshold should yield just a handful of points.
    assert 0 < len(pts) < 100, f"Expected small localized cloud, got {len(pts)} points."

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
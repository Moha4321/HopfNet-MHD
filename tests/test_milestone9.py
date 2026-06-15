"""
Milestone 9: Magnetic Null Finder unit tests.

Verifies the Haynes-Parnell sub-grid Newton-Raphson iteration and 
Eigenvalue classification (Radial vs Spiral nulls).
"""
import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
import hopfnet.hopfnet_cpp as eng

N = 16
L = 2 * np.pi

def test_null_finder_radial_and_spiral():
    x = np.arange(N) * (L / N) - L / 2.0
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    # ---------------------------------------------------------
    # TEST 1: Radial Null (All real eigenvalues)
    # B = (x - 0.3, y + 0.1, -2z + 0.4)
    # div(B) = 1 + 1 - 2 = 0
    # Null exactly at (0.3, -0.1, 0.2)
    # ---------------------------------------------------------
    Bx1 = X - 0.3
    By1 = Y + 0.1
    Bz1 = -2 * Z + 0.4
    
    pos1, types1 = eng.find_nulls(N, L, Bx1, By1, Bz1)
    
    # Filter out periodic boundary ghosts caused by the non-periodic test field
    mask1 = np.linalg.norm(pos1, axis=1) < 2.0
    pos1 = pos1[mask1]
    types1 = types1[mask1]
    
    assert len(pos1) == 1, f"Expected 1 radial null, found {len(pos1)}"
    
    # Verify sub-grid Newton-Raphson precision
    np.testing.assert_allclose(pos1[0], [0.3, -0.1, 0.2], atol=1e-6)
    
    # Verify Classification: J = diag(1, 1, -2) -> All real -> Type 0
    assert types1[0] == 0, "Failed to classify Radial Null!"


    # ---------------------------------------------------------
    # TEST 2: Spiral Null (1 real, 2 complex conjugate eigenvalues)
    # This is the reconnection-driving topology!
    # B = (-x + 2y, -2x - y, 2z)
    # div(B) = -1 - 1 + 2 = 0
    # Null exactly at (0.0, 0.0, 0.0)
    # ---------------------------------------------------------
    Bx2 = -X + 2*Y
    By2 = -2*X - Y
    Bz2 = 2*Z
    
    pos2, types2 = eng.find_nulls(N, L, Bx2, By2, Bz2)
    
    # Filter out periodic boundary ghosts
    mask2 = np.linalg.norm(pos2, axis=1) < 2.0
    pos2 = pos2[mask2]
    types2 = types2[mask2]
    
    assert len(pos2) == 1, f"Expected 1 spiral null, found {len(pos2)}"
    np.testing.assert_allclose(pos2[0], [0.0, 0.0, 0.0], atol=1e-6)
    
    # Verify Classification: 
    # J = [[-1, 2, 0], [-2, -1, 0], [0, 0, 2]]
    # Eigenvalues: 2, and -1 +- 2i -> Has complex pairs -> Type 1
    assert types2[0] == 1, "Failed to classify Spiral Null!"

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
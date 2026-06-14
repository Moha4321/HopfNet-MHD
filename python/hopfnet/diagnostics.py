"""
Milestone 6: Physics Diagnostics

Evaluates the absolute topological invariants and thermodynamic quantities
of the Hall-MHD system to track conservation over time.
"""
import numpy as np

from . import hopfnet_cpp as eng
from .rhs import to_real, compute_B_and_J

def compute_diagnostics(grid, fft, A_hat, v_hat, d_i):
    """
    Computes system invariants: 
      - Max divergence of A and v (spectral precision check)
      - Kinetic, Magnetic, and Total Energy
      - Generalized Helicity (Hall-MHD topological invariant)
    """
    # 1. Divergence checks (k . A_hat and k . v_hat)
    KX = grid.kx()[:, None, None]
    KY = grid.ky()[None, :, None]
    KZ = grid.kz()[None, None, :]
    
    k_dot_A = KX * A_hat[0] + KY * A_hat[1] + KZ * A_hat[2]
    div_A_max = np.max(np.abs(k_dot_A))
    
    k_dot_v = KX * v_hat[0] + KY * v_hat[1] + KZ * v_hat[2]
    div_v_max = np.max(np.abs(k_dot_v))

    # 2. Transform to Real Space
    A_real = to_real(fft, A_hat)
    v_real = to_real(fft, v_hat)

    # Compute Magnetic Field
    B_hat, _ = compute_B_and_J(grid, A_hat)
    B_real = to_real(fft, B_hat)

    # Compute Fluid Vorticity (omega = curl v)
    omega_hat = eng.spectral_curl(grid, *v_hat)
    omega_real = to_real(fft, omega_hat)

    # 3. Integration volume element
    dx = grid.L / grid.N
    dV = dx**3

    # 4. Energy Computation
    v_sq = v_real[0]**2 + v_real[1]**2 + v_real[2]**2
    B_sq = B_real[0]**2 + B_real[1]**2 + B_real[2]**2
    
    E_k = 0.5 * np.sum(v_sq) * dV
    E_m = 0.5 * np.sum(B_sq) * dV
    E_total = E_k + E_m

    # 5. Generalized Helicity (Eq. 41)
    # A* = A + d_i * v
    # B* = B + d_i * omega
    A_star = tuple(A_real[i] + d_i * v_real[i] for i in range(3))
    B_star = tuple(B_real[i] + d_i * omega_real[i] for i in range(3))
    
    # H_gen = \int (A* . B*) dV
    h_density = A_star[0]*B_star[0] + A_star[1]*B_star[1] + A_star[2]*B_star[2]
    H_gen = np.sum(h_density) * dV

    return {
        "div_A_max": div_A_max,
        "div_v_max": div_v_max,
        "E_k": E_k,
        "E_m": E_m,
        "E_total": E_total,
        "H_gen": H_gen
    }
r"""
Milestone 10: Topological Diagnostics

Calculates global topological invariants including standard Magnetic Helicity,
the Moffatt Linking Number, the Magnetic Flux, the Isotropic Energy Spectrum,
and the Generalized Helicity decay rate.
"""
import numpy as np
from . import hopfnet_cpp as eng
from .rhs import to_real, compute_B_and_J

def compute_magnetic_helicity(grid, fft, A_hat, B_hat=None):
    r"""Computes standard Magnetic Helicity H = \int A . B dV."""
    if B_hat is None:
        B_hat, _ = compute_B_and_J(grid, A_hat)
        
    A_real = to_real(fft, A_hat)
    B_real = to_real(fft, B_hat)
    
    dV = (grid.L / grid.N)**3
    H = np.sum(A_real[0]*B_real[0] + A_real[1]*B_real[1] + A_real[2]*B_real[2]) * dV
    return H

def compute_flux(grid, fft, B_hat, R=1.0, d=0.3):
    r"""
    Computes the magnetic flux \Phi through Ring 1 of the Hopf link.
    Ring 1 sits in the xy-plane (z=0), centered at (0, d, 0).

    IMPORTANT: This function is physically meaningful ONLY at t=0 (the initial
    Hopf link configuration). As the plasma evolves, the flux tubes deform and
    the ring geometry is no longer well-defined. Always compute Phi0 once at
    t=0 and cache it. Never call this function at t > 0 and pass the result
    into compute_linking_number — use the stored Phi0 and calib_factor instead.
    """
    B_real = to_real(fft, B_hat)
    Bz = B_real[2]
    
    N = grid.N
    dx = grid.L / N
    x = np.arange(N) * dx - grid.L / 2.0
    X, Y = np.meshgrid(x, x, indexing='ij')
    
    # Locate the z=0 midplane index
    z_idx = N // 2
    
    # Mask out the disk defining the interior of the ring
    mask = (X**2 + (Y - d)**2) <= R**2
    
    # Flux = \int B_z dA
    dA = dx**2
    Phi = np.sum(Bz[:, :, z_idx][mask]) * dA
    return Phi

def compute_linking_number(grid, fft, A_hat, B_hat=None, Phi=None, calib_factor=1.0):
    r"""
    Computes the calibrated Moffatt Linking Number:
        Lk(t) = [H(t) / (2 * Phi_0^2)] * calib_factor

    where H(t) = \int A(t) . B(t) dV  (time-varying magnetic helicity)
    and   Phi_0 is the INITIAL magnetic flux through Ring 1 (fixed at t=0).

    Providing calib_factor = (2 * Phi_0^2) / H_0 normalizes so that Lk(t=0) = 1.0
    exactly, accounting for finite-core regularization of the Hopf link.

    Call pattern:
      At t=0 (compute calib_factor):
        Phi0 = compute_flux(grid, fft, B_hat_init)
        H0   = compute_magnetic_helicity(grid, fft, A_hat_init)
        calib = (2*Phi0**2) / H0
      At t>0 (track Lk):
        Lk = compute_linking_number(grid, fft, A_hat, B_hat,
                                    Phi=Phi0, calib_factor=calib)

    CRITICAL: Always pass Phi=Phi_0 (the INITIAL flux). Never pass the
    current-time flux — the ring geometry is undefined after reconnection.
    """
    if B_hat is None:
        B_hat, _ = compute_B_and_J(grid, A_hat)

    H = compute_magnetic_helicity(grid, fft, A_hat, B_hat)

    # At t=0 to bootstrap: compute Phi from current field (valid only then)
    if Phi is None:
        Phi = compute_flux(grid, fft, B_hat)

    if np.abs(Phi) < 1e-12:
        return 0.0

    # Core formula: Moffatt (1969), Lk = H / (2*Phi_0^2), scaled by calib_factor
    # so that Lk(t=0) = 1.0 exactly for a regularized finite-core Hopf link.
    return (H / (2.0 * Phi**2)) * calib_factor

def compute_1d_spectrum(grid, B_hat):
    r"""
    Computes the 1D isotropic energy spectrum E(k).
    Normalized via Parseval's Theorem so that sum(E_k) == E_magnetic.
    """
    energy_3d = 0.5 * (np.abs(B_hat[0])**2 + np.abs(B_hat[1])**2 + np.abs(B_hat[2])**2)
    
    k_mag = np.sqrt(grid.k2())
    k_bins = np.arange(0, int(np.max(k_mag)) + 2)
    E_k = np.zeros(len(k_bins) - 1)
    
    # Account for rFFT Hermitian redundancy
    weight = np.ones_like(energy_3d)
    weight[:, :, 1:-1] = 2.0 
    
    # Parseval normalization factor
    volume_factor = (grid.L**3) / (grid.N**6)
    weighted_energy = energy_3d * weight * volume_factor
    
    k_indices = np.digitize(k_mag, k_bins) - 1
    
    for i in range(len(E_k)):
        mask = (k_indices == i)
        E_k[i] = np.sum(weighted_energy[mask])
        
    return k_bins[:-1], E_k

def compute_helicity_decay_rate(H_gen_current, H_gen_prev, dt, diag_interval):
    r"""Computes the finite-difference time derivative dH_gen / dt."""
    return (H_gen_current - H_gen_prev) / (dt * diag_interval)
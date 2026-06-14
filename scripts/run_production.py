"""
Milestone 8: 128^3 Production Run & Spectral Analysis

Runs the high-resolution Hall-MHD simulation and computes 
the 1D isotropic magnetic energy spectrum E(k).
"""
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
from hopfnet.simulate import HopfNetSimulation
from hopfnet.rhs import compute_B_and_J

def compute_1d_spectrum(grid, B_hat):
    """
    Computes the 1D isotropic energy spectrum E(k) by binning 
    the 3D spectral energy into 1D wavenumber shells.
    """
    N = grid.N
    # Multiply by 0.5 for magnetic energy: 0.5 * |B_hat|^2
    energy_3d = 0.5 * (np.abs(B_hat[0])**2 + np.abs(B_hat[1])**2 + np.abs(B_hat[2])**2)
    
    k2 = grid.k2()
    k_mag = np.sqrt(k2)
    
    # Create integer bins for k
    k_bins = np.arange(0, int(np.max(k_mag)) + 2)
    E_k = np.zeros(len(k_bins) - 1)
    
    # We must account for the rFFT Hermitian symmetry!
    # Modes with kz > 0 are counted twice (positive and negative z)
    # kz = 0 and kz = N/2 are counted once.
    weight = np.ones_like(energy_3d)
    weight[:, :, 1:-1] = 2.0 

    weighted_energy = energy_3d * weight
    
    # Bin the energy into shells
    k_indices = np.digitize(k_mag, k_bins) - 1
    
    for i in range(len(E_k)):
        mask = (k_indices == i)
        E_k[i] = np.sum(weighted_energy[mask])
        
    return k_bins[:-1], E_k

def run_production():
    print("==================================================")
    print(" HOPFNET-MHD: PHASE 1 PRODUCTION RUN (128^3) ")
    print("==================================================")
    
    # 128^3 grid pushes the fluid into a turbulent regime.
    # We use low resistivity to allow the Hopf link to twist and cascade.
    N_res = 128
    sim = HopfNetSimulation(N=N_res, dt=0.005, d_i=0.1, 
                            eta=5e-5, eta4=1e-6, 
                            nu=5e-5, nu4=1e-6, 
                            out_dir="production_128_out")
    
    # Run for 100 steps to allow the cascade to form
    sim.run(steps=100, diag_interval=10, save_interval=100)
    
    print("\nComputing 1D Magnetic Energy Spectrum...")
    B_hat, _ = compute_B_and_J(sim.grid, sim.A_hat)
    k_vals, E_k = compute_1d_spectrum(sim.grid, B_hat)
    
    # Save the spectrum to disk
    np.savez("production_128_out/spectrum_final.npz", k=k_vals, E_k=E_k)
    print("Spectrum saved to production_128_out/spectrum_final.npz")
    print("PRODUCTION RUN COMPLETE.")

if __name__ == "__main__":
    run_production()
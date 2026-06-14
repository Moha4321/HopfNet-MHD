"""
Milestone 6 & 7: The Main Simulation Orchestrator

Initializes the Spectral Grid, Hopf Link, and ETDRK4 Integrator,
and manages the main time-stepping loop with checkpointing.
"""
import os
import numpy as np

from . import hopfnet_cpp as eng
from .rhs import to_hat, to_real
from .etdrk4 import ETDRK4Integrator
from .diagnostics import compute_diagnostics

class HopfNetSimulation:
    def __init__(self, N=64, L=2*np.pi, dt=0.01,
                 d_i=0.1, eta=1e-3, eta4=1e-4, nu=1e-3, nu4=1e-4,
                 out_dir="data_out"):
        self.N = N
        self.L = L
        self.dt = dt
        self.d_i = d_i
        self.out_dir = out_dir
        
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        print(f"Initializing SpectralGrid and FFT (N={N})...")
        self.grid = eng.SpectralGrid(N, L)
        self.fft = eng.FFT3D(N)
        
        print("Precomputing ETDRK4 Integrator coefficients...")
        self.integrator = ETDRK4Integrator(self.grid, self.fft, dt, d_i, eta, eta4, nu, nu4)

        print("Generating Regularized Magnetic Hopf Link...")
        Ax, Ay, Az = eng.compute_hopf_link(N, L, R=1.0, d=0.3, a_core=0.2, I0=1.0, mu0=1.0, n_quad=64)
        A_hat = to_hat(self.fft, (Ax, Ay, Az))
        self.A_hat = eng.project_field(self.grid, *A_hat)

        print("Initializing velocity field (plasma at rest)...")
        v_zero = np.zeros((N, N, N))
        self.v_hat = to_hat(self.fft, (v_zero, v_zero, v_zero))
        
        self.step_num = 0
        self.history = []

    def run(self, steps, diag_interval=10, save_interval=50):
        print(f"Starting integration for {steps} steps...")
        for i in range(steps):
            if self.step_num % diag_interval == 0:
                diags = compute_diagnostics(self.grid, self.fft, self.A_hat, self.v_hat, self.d_i)
                diags["step"] = self.step_num
                diags["time"] = self.step_num * self.dt
                self.history.append(diags)
                
                print(f"Step {self.step_num:04d} | t={diags['time']:05.3f} | "
                      f"E_tot: {diags['E_total']:.6e} | H_gen: {diags['H_gen']:.6e} | "
                      f"div_A: {diags['div_A_max']:.2e}")

            if self.step_num % save_interval == 0:
                self.save_checkpoint()

            # Advance state exactly one timestep
            self.A_hat, self.v_hat = self.integrator.step(self.A_hat, self.v_hat)
            self.step_num += 1

        # Force a final diagnostic and checkpoint at the end of the run
        diags = compute_diagnostics(self.grid, self.fft, self.A_hat, self.v_hat, self.d_i)
        diags["step"] = self.step_num
        diags["time"] = self.step_num * self.dt
        self.history.append(diags)
        print("Integration complete.")
        return self.history

    def save_checkpoint(self):
        A_real = to_real(self.fft, self.A_hat)
        v_real = to_real(self.fft, self.v_hat)
        
        path = os.path.join(self.out_dir, f"checkpoint_{self.step_num:04d}.npz")
        np.savez(path, 
                 step=self.step_num, 
                 time=self.step_num * self.dt,
                 A_x=A_real[0], A_y=A_real[1], A_z=A_real[2],
                 v_x=v_real[0], v_y=v_real[1], v_z=v_real[2])
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
from hopfnet.simulate import HopfNetSimulation
from hopfnet import hopfnet_cpp as eng
from hopfnet.rhs import to_hat, to_real, compute_B_and_J
from hopfnet.topology import compute_magnetic_helicity, compute_flux, compute_linking_number

def generate_abc_flow(grid, fft, V0=0.6, k0=2.0):
    """
    The Arnold-Beltrami-Childress (ABC) Flow.
    A deterministic, perfectly divergence-free, maximally helical fluid flow
    used in dynamo theory to forcefully fold and twist magnetic fields.
    """
    N = grid.N
    L = grid.L
    
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    # ABC Flow equations
    vx = V0 * (np.sin(k0 * Z) + np.cos(k0 * Y))
    vy = V0 * (np.sin(k0 * X) + np.cos(k0 * Z))
    vz = V0 * (np.sin(k0 * Y) + np.cos(k0 * X))
    
    v_hat = to_hat(fft, (vx, vy, vz))
    v_hat_proj = eng.project_field(grid, *v_hat)
    
    return v_hat_proj

def run_production():
    # ---------------------------------------------------------
    # POSTDOCTORAL PHYSICS PARAMETERS (ABC-Driven Collapse)
    # ---------------------------------------------------------
    N = 128
    L = 2 * np.pi
    dt = 0.005          
    
    R_ring = 1.0
    d_offset = 0.85     
    a_core = 0.12       
    
    # Accelerated Reconnection Parameters
    d_i = 0.25          # Widened Hall region for faster ion decoupling
    eta = 2e-3          # Increased baseline resistivity
    nu  = 5e-4          # Pm = 0.25 (Fluid evacuates X-point easily)
    
    eta4 = 2e-5         
    nu4  = 2e-5         
    
    V_abc = 0.6         # Aggressive ABC Flow amplitude
    # ---------------------------------------------------------

    print("============================================================")
    print(" PHASE 2: ABC-DRIVEN TOPOLOGICAL COLLAPSE (128^3)")
    print(f" eta={eta}, nu={nu}, d_i={d_i}, a_core={a_core}")
    print(f" Driving Mechanism: ABC Flow (V0={V_abc}, k=2)")
    print("============================================================")

    sim = HopfNetSimulation(N=N, L=L, dt=dt, d_i=d_i, 
                            eta=eta, eta4=eta4, nu=nu, nu4=nu4, 
                            out_dir="production_128_out")
    
    print(f"Generating Intimate Hopf Link...")
    Ax, Ay, Az = eng.compute_hopf_link(N, L, R=R_ring, d=d_offset, a_core=a_core, I0=1.0, mu0=1.0, n_quad=64)
    sim.A_hat = eng.project_field(sim.grid, *to_hat(sim.fft, (Ax, Ay, Az)))

    print("Injecting ABC Beltrami Flow...")
    sim.v_hat = generate_abc_flow(sim.grid, sim.fft, V0=V_abc)

    B_hat, _ = compute_B_and_J(sim.grid, sim.A_hat)
    Phi0 = compute_flux(sim.grid, sim.fft, B_hat, R=R_ring, d=d_offset)
    H0 = compute_magnetic_helicity(sim.grid, sim.fft, sim.A_hat, B_hat)
    calib = (2.0 * Phi0**2) / H0
    
    print(f"t=0 | H0={H0:.4e} | Phi0={Phi0:.4e} | calib={calib:.4f}")

    steps = 400  # Will collapse beautifully within t=2.0
    diag_interval = 10
    
    for step in range(steps):
        if step % diag_interval == 0:
            B_hat, J_hat = compute_B_and_J(sim.grid, sim.A_hat)
            Lk = compute_linking_number(sim.grid, sim.fft, sim.A_hat, B_hat, Phi=Phi0, calib_factor=calib)
            
            B_real = to_real(sim.fft, B_hat)
            nulls, types = eng.find_nulls(N, L, B_real[0], B_real[1], B_real[2])
            
            cloud = eng.extract_point_cloud(sim.grid, sim.fft, J_hat[0], J_hat[1], J_hat[2], 0.6)
            
            t = step * dt
            print(f"Step {step:04d} | t={t:.3f} | Lk={Lk:.4f} | Nulls={len(nulls)} | Active Tear Points={cloud.shape[0]}")
            
        sim.A_hat, sim.v_hat = sim.integrator.step(sim.A_hat, sim.v_hat)

if __name__ == "__main__":
    run_production()
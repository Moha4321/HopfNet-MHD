import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
from hopfnet.simulate import HopfNetSimulation
from hopfnet import hopfnet_cpp as eng
from hopfnet.rhs import to_hat, to_real
from hopfnet.topology import compute_magnetic_helicity, compute_flux, compute_linking_number

def generate_driven_inflow(grid, fft, V0=0.5):
    """
    Generates a macroscopic, divergence-free fluid flow that mechanically 
    drives the two Hopf rings together along the y-axis, mimicking 
    the macroscopic inflow of a Sweet-Parker reconnection layer.
    """
    N = grid.N
    L = grid.L
    
    # 1D coordinates matching the C++ grid
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    # Stagnation-like flow: Pushes in along Y, evacuates along X and Z
    # We use sin(2*pi*Y/L) so it's strictly periodic and pushes Y>0 down, Y<0 up.
    vx =  V0 * np.sin(2 * np.pi * X / L) * np.cos(2 * np.pi * Y / L)
    vy = -V0 * np.cos(2 * np.pi * X / L) * np.sin(2 * np.pi * Y / L)
    vz = np.zeros_like(X)
    
    # Transform to spectral space
    v_hat = to_hat(fft, (vx, vy, vz))
    
    # Strictly enforce div(v) = 0 via the Coulomb projection tensor
    # This mathematically guarantees we don't inject acoustic shockwaves
    v_hat_proj = eng.project_field(grid, *v_hat)
    
    return v_hat_proj

def run_production():
    # ---------------------------------------------------------
    # POSTDOCTORAL PHYSICS PARAMETERS (GEM Reconnection Scaling)
    # ---------------------------------------------------------
    N = 128
    L = 2 * np.pi
    dt = 0.005          # CFL constrained by explicit advection/Hall
    
    d_i = 0.15          # Hall scale (ions decouple here)
    a_core = 0.10       # Tube thickness (strictly less than d_i)
    
    eta = 1e-3          # Magnetic diffusivity
    nu  = 1e-4          # Kinematic viscosity (Pm = 0.1 -> fluid flows easily)
    
    eta4 = 1e-5         # Hyper-resistivity (captures sub-grid tearing)
    nu4  = 1e-5         # Hyper-viscosity
    
    V0 = 0.3            # Driving inflow velocity amplitude
    # ---------------------------------------------------------

    print("============================================================")
    print(" PHASE 2 PHYSICS VERIFICATION RUN (128^3) - DRIVEN RECONNECTION")
    print(f" eta={eta}, nu={nu}, d_i={d_i}, a_core={a_core}, V_inflow={V0}")
    print("============================================================")

    # Initialize the simulation orchestrator
    sim = HopfNetSimulation(N=N, L=L, dt=dt, d_i=d_i, 
                            eta=eta, eta4=eta4, nu=nu, nu4=nu4, 
                            out_dir="production_128_out")
    
    # Re-initialize the Hopf link with tighter cores
    print(f"Generating Tight-Core Hopf Link (a_core={a_core})...")
    Ax, Ay, Az = eng.compute_hopf_link(N, L, R=1.0, d=0.3, a_core=a_core, I0=1.0, mu0=1.0, n_quad=64)
    sim.A_hat = eng.project_field(sim.grid, *to_hat(sim.fft, (Ax, Ay, Az)))

    # Inject the driven reconnection inflow
    print("Injecting incompressible driven inflow...")
    sim.v_hat = generate_driven_inflow(sim.grid, sim.fft, V0=V0)

    # Calculate initial invariants for the calibrator
    B_hat, _ = eng.spectral_curl(sim.grid, *sim.A_hat)
    Phi0 = compute_flux(sim.grid, sim.fft, B_hat, R=1.0, d=0.3)
    H0 = compute_magnetic_helicity(sim.grid, sim.fft, sim.A_hat, B_hat)
    calib = (2.0 * Phi0**2) / H0
    
    print(f"t=0 | H0={H0:.4e} | Phi0={Phi0:.4e} | calib={calib:.4f}")

    # We will run for 1000 steps (t = 5.0) to give it time to crash the tubes together
    steps = 1000
    diag_interval = 10
    
    for step in range(steps):
        # Diagnostics
        if step % diag_interval == 0:
            B_hat, J_hat = eng.spectral_curl(sim.grid, *sim.A_hat)
            Lk = compute_linking_number(sim.grid, sim.fft, sim.A_hat, B_hat, Phi=Phi0, calib_factor=calib)
            
            # Find magnetic null points
            B_real = to_real(sim.fft, B_hat)
            nulls, types = eng.find_nulls(N, L, B_real[0], B_real[1], B_real[2])
            
            # Extract point cloud for the active tearing regions (J > 0.6 J_max)
            # This is exactly what the AI will ingest in Phase 3
            cloud = eng.extract_point_cloud(sim.grid, sim.fft, J_hat[0], J_hat[1], J_hat[2], 0.6)
            
            t = step * dt
            print(f"Step {step:04d} | t={t:.3f} | Lk={Lk:.4f} | Nulls={len(nulls)} | Active Tear Points={cloud.shape[0]}")
            
        # Time integration via ETDRK4
        sim.A_hat, sim.v_hat = sim.integrator.step(sim.A_hat, sim.v_hat)

if __name__ == "__main__":
    run_production()
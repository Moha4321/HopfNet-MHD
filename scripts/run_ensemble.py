"""
Milestone 12: Phase 2 Ensemble Generator

Generates a 40-run ensemble of Hall-MHD selective decay.
Extracts Point Cloud time series (shape operators of current sheets) 
and topological Reconnection Onset (t_c) labels via the Moffatt Linking Number.
"""
import sys
import os
import json
import numpy as np
from scipy.stats.qmc import LatinHypercube

# Ensure Python can find the local hopfnet module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

from hopfnet.simulate import HopfNetSimulation
from hopfnet.rhs import compute_B_and_J
from hopfnet.topology import compute_linking_number, compute_magnetic_helicity, compute_flux
from hopfnet import hopfnet_cpp as eng

def setup_sampler(n_samples=40, seed=42):
    """Generates strictly stratified LHS parameters for physics sweeps."""
    sampler = LatinHypercube(d=2, seed=seed)
    samples = sampler.random(n=n_samples)
    
    # eta controls diffusion scale, a_core controls initial gradient steepness
    etas = 5e-4 + samples[:, 0] * (5e-3 - 5e-4)
    a_cores = 0.1 + samples[:, 1] * (0.4 - 0.1)
    return etas, a_cores

def run_single_simulation(run_id, eta, a_core, N=128, steps=200, diag_interval=5, out_dir="data_ensemble"):
    print(f"\n--- Starting Run {run_id:03d} | eta={eta:.2e}, a_core={a_core:.3f} ---")

    checkpoint_dir = os.path.join(out_dir, f"run_{run_id:03d}_checkpoints")
    sim = HopfNetSimulation(N=N, dt=0.005, eta=eta, nu=eta, d_i=0.1, out_dir=checkpoint_dir)

    Ax, Ay, Az = eng.compute_hopf_link(N, sim.L, R=1.0, d=0.3, a_core=a_core, I0=1.0, mu0=1.0, n_quad=64)
    A_hat_raw = (sim.fft.forward(Ax), sim.fft.forward(Ay), sim.fft.forward(Az))
    sim.A_hat = eng.project_field(sim.grid, *A_hat_raw)

    # Compute Phi0 and calib_factor ONCE at t=0 from the initial field.
    # Phi0 is cached and passed to every subsequent Lk call — never recomputed
    # from the evolving B_hat, since the ring geometry is undefined after reconnection.
    B_hat_init, _ = compute_B_and_J(sim.grid, sim.A_hat)
    H0   = compute_magnetic_helicity(sim.grid, sim.fft, sim.A_hat, B_hat_init)
    Phi0 = compute_flux(sim.grid, sim.fft, B_hat_init)
    calib_factor = (2.0 * Phi0**2) / H0 if np.abs(H0) > 1e-12 else 1.0

    point_clouds = []
    time_series  = []
    t_c          = None

    for step in range(steps):
        time_now = step * sim.dt

        if step % diag_interval == 0:
            B_hat, J_hat = compute_B_and_J(sim.grid, sim.A_hat)

            # Pass Phi0 (initial flux) explicitly — physically correct for t > 0
            Lk = compute_linking_number(sim.grid, sim.fft, sim.A_hat, B_hat,
                                        Phi=Phi0, calib_factor=calib_factor)

            if t_c is None and Lk < 0.5:
                t_c = time_now
                print(f"[*] TOPOLOGICAL UNLINKING DETECTED at t_c = {t_c:.3f} (Step {step})")

            pc = eng.extract_point_cloud(sim.grid, sim.fft,
                                          J_hat[0], J_hat[1], J_hat[2], threshold=0.6)
            point_clouds.append(pc)
            time_series.append(time_now)

            print(f"Step {step:03d} | t={time_now:.3f} | Lk: {Lk:.3f} | PC size: {len(pc)}")

        # Step physics via ETDRK4
        sim.A_hat, sim.v_hat = sim.integrator.step(sim.A_hat, sim.v_hat)

    # Save final dataset pair (X, y) -> (Point Clouds, t_c)
    np.savez_compressed(
        os.path.join(out_dir, f"dataset_{run_id:03d}.npz"),
        point_clouds=np.array(point_clouds, dtype=object),
        time_series=np.array(time_series),
        t_c=t_c if t_c is not None else -1.0,
        eta=eta,
        a_core=a_core
    )
    print(f"Saved dataset_{run_id:03d}.npz")
    return t_c

def main():
    out_dir = "data_ensemble"
    os.makedirs(out_dir, exist_ok=True)
    status_file = os.path.join(out_dir, "status.json")
    
    etas, a_cores = setup_sampler(40)
    
    # Load status for resumability (prevents losing progress if stopped)
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            status = json.load(f)
    else:
        status = {}

    for i in range(40):
        run_key = f"run_{i:03d}"
        if run_key in status and status[run_key].get("completed", False):
            print(f"Skipping {run_key}, already completed.")
            continue

        # Runs that previously did not reconnect within t=1.0 (200 steps) are
        # almost certainly low-eta cases on longer reconnection timescales.
        # Extend to 600 steps (t_max=3.0) to capture Sweet-Parker scaling.
        prior_t_c = status.get(run_key, {}).get("t_c", None)
        n_steps = 600 if (prior_t_c is None or prior_t_c < 0) else 200

        t_c = run_single_simulation(i, etas[i], a_cores[i],
                                     N=128, steps=n_steps,
                                     diag_interval=5, out_dir=out_dir)

        status[run_key] = {"completed": True, "t_c": t_c,
                           "eta": float(etas[i]), "a_core": float(a_cores[i])}
        with open(status_file, "w") as f:
            json.dump(status, f, indent=4)

if __name__ == "__main__":
    main()
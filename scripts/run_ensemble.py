"""
Milestone 12: Phase 2 Ensemble Generator

Generates a 40-run ensemble of Hall-MHD selective decay.
Extracts Point Cloud time series and Reconnection Onset (t_c) labels.
"""
import sys
import os
import json
import numpy as np
from scipy.stats.qmc import LatinHypercube

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
from hopfnet.simulate import HopfNetSimulation
from hopfnet.rhs import compute_B_and_J, to_real
from hopfnet import hopfnet_cpp as eng

def setup_sampler(n_samples=40, seed=42):
    """Generates strictly stratified LHS parameters."""
    sampler = LatinHypercube(d=2, seed=seed)
    samples = sampler.random(n=n_samples)
    
    etas = 5e-4 + samples[:, 0] * (5e-3 - 5e-4)
    a_cores = 0.1 + samples[:, 1] * (0.4 - 0.1)
    return etas, a_cores

def run_single_simulation(run_id, eta, a_core, N=128, steps=200, diag_interval=5, out_dir="data_ensemble"):
    print(f"\n--- Starting Run {run_id:03d} | eta={eta:.2e}, a_core={a_core:.3f} ---")
    
    sim = HopfNetSimulation(N=N, dt=0.005, eta=eta, nu=eta, d_i=0.1, out_dir=os.path.join(out_dir, f"run_{run_id:03d}_checkpoints"))
    
    # We must override the initialization to use the sampled a_core
    Ax, Ay, Az = eng.compute_hopf_link(N, sim.L, R=1.0, d=0.3, a_core=a_core, I0=1.0, mu0=1.0, n_quad=64)
    sim.A_hat = eng.project_field(sim.grid, *sim.rhs_to_hat((Ax, Ay, Az))) if hasattr(sim, 'rhs_to_hat') else eng.project_field(sim.grid, *[sim.fft.forward(a) for a in (Ax, Ay, Az)])

    point_clouds = []
    time_series = []
    t_c = None

    for step in range(steps):
        time_now = step * sim.dt
        
        # Diagnostic Step: Check for Nulls and Extract Point Cloud
        if step % diag_interval == 0:
            B_hat, J_hat = compute_B_and_J(sim.grid, sim.A_hat)
            B_real = [sim.fft.inverse(b) for b in B_hat]
            
            # 1. Null Finder
            pos, types = eng.find_nulls(N, sim.L, B_real[0], B_real[1], B_real[2])
            spiral_nulls = np.sum(types == 1)
            
            # If t_c is not set, and we found a spiral null, RECORD RECONNECTION ONSET!
            if t_c is None and spiral_nulls > 0:
                t_c = time_now
                print(f"[*] RECONNECTION ONSET DETECTED at t_c = {t_c:.3f} (Step {step})")
            
            # 2. Point Cloud Extraction (Shape Operator of Current Sheets)
            pc = eng.extract_point_cloud(sim.grid, sim.fft, J_hat[0], J_hat[1], J_hat[2], threshold=0.6)
            
            point_clouds.append(pc)
            time_series.append(time_now)
            
            print(f"Step {step:03d} | t={time_now:.3f} | Nulls: {len(pos)} (Spiral: {spiral_nulls}) | PC size: {len(pc)}")

        # Step physics
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
    
    # Load status for resumability
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
            
        t_c = run_single_simulation(i, etas[i], a_cores[i], N=128, steps=150, diag_interval=5, out_dir=out_dir)
        
        status[run_key] = {"completed": True, "t_c": t_c, "eta": etas[i], "a_core": a_cores[i]}
        with open(status_file, "w") as f:
            json.dump(status, f, indent=4)

if __name__ == "__main__":
    main()
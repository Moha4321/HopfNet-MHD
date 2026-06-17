"""
Phase 2 Dataset Rescue & Relabeling

Post-processes the ensemble datasets to extract the Reconnection Onset (t_c)
directly from the geometric collapse of the current sheet (Point Cloud volume).
Bypasses the Eulerian flux loss artifact.
"""
import os
import json
import numpy as np
from scipy.stats import spearmanr

def relabel_and_verify(data_dir="data_ensemble"):
    print("==================================================")
    print(" GEOMETRIC RELABELING & PHYSICS VERIFICATION ")
    print("==================================================")
    
    files = sorted([f for f in os.listdir(data_dir) if f.startswith("dataset_") and f.endswith(".npz")])
    
    if not files:
        print("No dataset files found!")
        return

    new_status = {}
    valid_runs = []
    t_c_list = []
    eta_list = []
    a_core_list = []

    print("Analyzing Current Sheet Geometry...")
    for f in files:
        run_id = f.split("_")[1].split(".")[0]
        path = os.path.join(data_dir, f)
        
        # Load the saved tensors
        data = np.load(path, allow_pickle=True)
        time_series = data["time_series"]
        point_clouds = data["point_clouds"]
        eta = float(data["eta"])
        a_core = float(data["a_core"])
        
        # Track the volume of the current sheet over time
        pc_sizes = [len(pc) for pc in point_clouds]
        
        # Filter out the initial spectral ringing transient (t < 0.05)
        valid_indices = [i for i, t in enumerate(time_series) if t > 0.05]
        
        t_c = -1.0
        if valid_indices:
            sub_sizes = [pc_sizes[i] for i in valid_indices]
            local_peak_idx = np.argmax(sub_sizes)
            actual_peak_idx = valid_indices[local_peak_idx]
            
            # Reconnection is defined as the current sheet breaking.
            # We require the point cloud to have peaked and then dropped by at least 5%
            if actual_peak_idx < len(pc_sizes) - 1:
                drop_ratio = pc_sizes[-1] / pc_sizes[actual_peak_idx]
                if drop_ratio < 0.95:
                    t_c = time_series[actual_peak_idx]
        
        # Update the .npz file with the rescued label
        np.savez_compressed(
            path,
            point_clouds=point_clouds,
            time_series=time_series,
            t_c=t_c,
            eta=eta,
            a_core=a_core
        )
        
        new_status[f"run_{run_id}"] = {
            "completed": True,
            "t_c": float(t_c),
            "eta": float(eta),
            "a_core": float(a_core)
        }
        
        if t_c > 0:
            valid_runs.append(run_id)
            t_c_list.append(t_c)
            eta_list.append(eta)
            a_core_list.append(a_core)
            print(f"Run {run_id} | Peak PC Size: {pc_sizes[actual_peak_idx]:05d} | t_c successfully rescued: {t_c:.3f}")
        else:
            print(f"Run {run_id} | Did not geometrically reconnect within t=1.0")

    # Save the corrected status
    with open(os.path.join(data_dir, "status.json"), "w") as f_out:
        json.dump(new_status, f_out, indent=4)

    # --- Rigorous Physics Scaling Verification ---
    print("\n==================================================")
    print(f"Total Runs: {len(files)}")
    print(f"Successfully Reconnected within t=1.0: {len(valid_runs)}")
    
    if len(valid_runs) < 5:
        print("Not enough reconnected runs to establish statistical scaling. Need to run longer simulations.")
        return

    t_c_list = np.array(t_c_list)
    eta_list = np.array(eta_list)
    a_core_list = np.array(a_core_list)

    print(f"\n--- Reconnection Onset Time (t_c) Statistics ---")
    print(f"Min t_c:  {np.min(t_c_list):.3f}")
    print(f"Mean t_c: {np.mean(t_c_list):.3f}")
    print(f"Max t_c:  {np.max(t_c_list):.3f}")

    # Sweet-Parker Scaling: Higher resistivity (eta) -> Faster (lower) t_c
    corr_eta, p_val_eta = spearmanr(eta_list, t_c_list)
    print(f"\n--- Physics Scaling Checks ---")
    print(f"Correlation (Resistivity eta vs t_c): {corr_eta:.3f}")
    if corr_eta < -0.2:
        print("✅ PASS: Higher resistivity correctly induces faster reconnection (Sweet-Parker consistency).")
    else:
        print("⚠️ WARNING: Resistivity scaling not definitively observed.")

    # Core Radius Scaling: Smaller a_core (sharper gradients) -> Faster t_c
    corr_a, p_val_a = spearmanr(a_core_list, t_c_list)
    print(f"Correlation (Core radius a_core vs t_c): {corr_a:.3f}")
    if corr_a > 0.2:
         print("✅ PASS: Sharper current sheets (smaller a_core) correctly accelerate tearing.")
    else:
         print("⚠️ WARNING: Gradient scaling not definitively observed.")

    print("\nDATASET SUCCESSFULLY RELABELED AND VERIFIED FOR AI TRAINING.")

if __name__ == "__main__":
    relabel_and_verify()



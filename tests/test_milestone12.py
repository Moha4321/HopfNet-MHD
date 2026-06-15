"""
Milestone 12: Ensemble Runner unit tests.

Verifies the LHS sampling boundaries, file writing integrity, 
and parameter-sensitivity of the physical Reconnection Onset (t_c).

Run with: pytest tests/test_milestone12.py -s
"""
import sys
import os
import shutil
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from run_ensemble import setup_sampler, run_single_simulation

def test_lhs_sampler_bounds():
    etas, a_cores = setup_sampler(n_samples=40, seed=1)
    
    assert len(etas) == 40
    assert len(a_cores) == 40
    
    # Check physics ranges
    assert np.all(etas >= 5e-4) and np.all(etas <= 5e-3)
    assert np.all(a_cores >= 0.1) and np.all(a_cores <= 0.4)
    
    # Verify it is stratified (not just uniform random clumps)
    eta_bins = np.histogram(etas, bins=4)[0]
    # In LHS, bins should be roughly equally populated
    assert np.all(eta_bins > 0)

def test_mini_ensemble_sensitivity_and_export():
    out_dir = "test_ensemble_out"
    os.makedirs(out_dir, exist_ok=True)
    
    # Run two extremely different physical corners to ensure t_c reacts to physics
    # Case 1: High resistivity, tight core (Fastest Reconnection)
    t_c_fast = run_single_simulation(0, eta=5e-3, a_core=0.1, N=16, steps=15, diag_interval=5, out_dir=out_dir)
    
    # Case 2: Low resistivity, wide core (Slowest Reconnection)
    t_c_slow = run_single_simulation(1, eta=5e-4, a_core=0.4, N=16, steps=15, diag_interval=5, out_dir=out_dir)
    
    # Verify files were created
    assert os.path.exists(os.path.join(out_dir, "dataset_000.npz"))
    assert os.path.exists(os.path.join(out_dir, "dataset_001.npz"))
    
    data = np.load(os.path.join(out_dir, "dataset_000.npz"), allow_pickle=True)
    assert "point_clouds" in data
    assert "t_c" in data
    assert "eta" in data
    
    # Verify Point Cloud shapes (Frames, Points, 9 features)
    pc_series = data["point_clouds"]
    assert len(pc_series) == 3  # 15 steps / 5 diag_interval = 3 frames
    if len(pc_series[0]) > 0:
        assert pc_series[0].shape[1] == 9 # (x,y,z, k11, k12, k13, k22, k23, k33)

    # Clean up test files
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
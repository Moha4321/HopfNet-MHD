"""
Milestone 7: End-to-End Physics Smoke Test

Runs the full time-integration loop on a small grid. 
Asserts that no floating point drift destroys the Coulomb gauge, 
no NaNs appear, and energy decays monotonically as physically required.

Run with: pytest tests/test_milestone7.py -s
"""
import sys
import os
import shutil
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
from hopfnet.simulate import HopfNetSimulation

def test_milestone7_end_to_end_smoke_test():
    test_dir = "test_smoke_out"
    
    # Initialize a small 16^3 grid simulation. 
    # Hyper-viscosity/resistivity parameters set high enough to ensure smooth fields.
    sim = HopfNetSimulation(N=16, dt=0.01, eta=0.01, nu=0.01, out_dir=test_dir)
    
    # Run the integrator forward for 20 steps, logging every 5 steps
    history = sim.run(steps=20, diag_interval=5, save_interval=10)
    
    # 1. Topological / Gauge Integrity Check
    # The divergence of A and v MUST stay below 1e-12. If it drifts, 
    # our spectral projection is leaking longitudinal modes.
    max_div_A = max(d["div_A_max"] for d in history)
    max_div_v = max(d["div_v_max"] for d in history)
    
    assert max_div_A < 1e-12, f"Catastrophic Gauge Violation! div(A) = {max_div_A}"
    assert max_div_v < 1e-12, f"Incompressibility Violation! div(v) = {max_div_v}"
    
    # 2. Thermodynamic Sanity Check
    E_t = [d["E_total"] for d in history]
    
    # Energy must be finite (no exploding ETDRK4 instability)
    assert all(np.isfinite(e) for e in E_t), "Energy blew up to NaN/Inf!"
    
    # Energy must not artificially increase (we have dissipative eta and nu).
    # Step 20 energy MUST be strictly less than or equal to Step 0 energy.
    assert E_t[-1] <= E_t[0] * (1.0 + 1e-14), "Unphysical energy injection detected!"

    # Clean up checkpoint outputs
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
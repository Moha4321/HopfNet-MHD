"""
Phase 2 Physics Verification Run

Single 128^3 run at low resistivity (eta=5e-4) and sharp core (a_core=0.15).
Expected physics:
  - Lk drops cleanly from 1.0 toward 0 by t ~ 0.5-1.5
  - A spiral null (Type 1) appears in the null finder near t_c
  - E(k) spectrum shows energy cascade developing

Expected timing on M4: ~55 minutes (400 steps x ~8 sec/step at 128^3)

Run with: python scripts/verify_physics_run.py
"""
import sys, os, time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

from hopfnet.simulate import HopfNetSimulation
from hopfnet.rhs import compute_B_and_J, to_real
from hopfnet.topology import (compute_magnetic_helicity, compute_flux,
                               compute_linking_number, compute_1d_spectrum)
from hopfnet import hopfnet_cpp as eng

def run_verification():
    print("=" * 60)
    print(" PHASE 2 PHYSICS VERIFICATION RUN (128^3)")
    print(" eta=5e-4, a_core=0.15, t_max=2.0 (400 steps)")
    print("=" * 60)

    N = 128
    out_dir = "verify_physics_out"
    os.makedirs(out_dir, exist_ok=True)

    sim = HopfNetSimulation(N=N, dt=0.005, eta=5e-4, nu=5e-4,
                            d_i=0.1, out_dir=out_dir)

    # Override a_core
    Ax, Ay, Az = eng.compute_hopf_link(N, sim.L, R=1.0, d=0.3, a_core=0.15,
                                        I0=1.0, mu0=1.0, n_quad=64)
    A_hat_raw = (sim.fft.forward(Ax), sim.fft.forward(Ay), sim.fft.forward(Az))
    sim.A_hat = eng.project_field(sim.grid, *A_hat_raw)

    # Bootstrap at t=0
    B_hat_init, _ = compute_B_and_J(sim.grid, sim.A_hat)
    H0   = compute_magnetic_helicity(sim.grid, sim.fft, sim.A_hat, B_hat_init)
    Phi0 = compute_flux(sim.grid, sim.fft, B_hat_init)
    calib = (2.0 * Phi0**2) / H0 if np.abs(H0) > 1e-12 else 1.0

    print(f"\nt=0 | H0={H0:.4e} | Phi0={Phi0:.4e} | calib={calib:.4f}")
    print(f"     Lk(t=0) should be exactly 1.0...")
    Lk0 = compute_linking_number(sim.grid, sim.fft, sim.A_hat, B_hat_init,
                                  Phi=Phi0, calib_factor=calib)
    print(f"     Lk(t=0) = {Lk0:.6f}  {'✅' if abs(Lk0 - 1.0) < 1e-6 else '❌ WRONG'}")

    # Storage
    history = []
    t_c = None
    null_t_c = None

    steps = 400
    diag_interval = 10
    t0_wall = time.time()

    for step in range(steps):
        t_now = step * sim.dt

        if step % diag_interval == 0:
            B_hat, J_hat = compute_B_and_J(sim.grid, sim.A_hat)

            # 1. Linking number (using fixed Phi0)
            Lk = compute_linking_number(sim.grid, sim.fft, sim.A_hat, B_hat,
                                         Phi=Phi0, calib_factor=calib)

            if t_c is None and Lk < 0.5:
                t_c = t_now
                print(f"\n  *** Lk t_c = {t_c:.3f} (step {step}) ***\n")

            # 2. Magnetic null finder (only after Lk starts dropping to save time)
            n_nulls = 0
            n_spiral = 0
            if Lk < 0.8 and null_t_c is None:
                B_real = to_real(sim.fft, B_hat)
                pos, types = eng.find_nulls(N, sim.L,
                                             B_real[0], B_real[1], B_real[2])
                n_nulls  = len(pos)
                n_spiral = int(np.sum(types == 1))
                if n_spiral > 0 and null_t_c is None:
                    null_t_c = t_now
                    print(f"  *** Spiral null t_c = {null_t_c:.3f} (step {step}), "
                          f"{n_spiral} spiral null(s) ***")

            # 3. Energy spectrum
            k_vals, E_k = compute_1d_spectrum(sim.grid, B_hat)
            E_tot = np.sum(E_k)

            elapsed = time.time() - t0_wall
            eta_remaining = (elapsed / (step + 1)) * (steps - step - 1)
            print(f"Step {step:04d} | t={t_now:.3f} | Lk={Lk:.4f} | "
                  f"E={E_tot:.4e} | nulls={n_nulls}(s={n_spiral}) | "
                  f"ETA: {eta_remaining/60:.1f} min")

            history.append({
                "step": step, "t": t_now, "Lk": Lk,
                "E_total": E_tot, "n_nulls": n_nulls, "n_spiral": n_spiral
            })

        sim.A_hat, sim.v_hat = sim.integrator.step(sim.A_hat, sim.v_hat)

    # Save results
    np.savez(os.path.join(out_dir, "verify_history.npz"),
             history=history, t_c_Lk=t_c, t_c_null=null_t_c,
             H0=H0, Phi0=Phi0)

    print("\n" + "=" * 60)
    print(" VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Lk-based t_c   : {t_c}")
    print(f"  Null-based t_c : {null_t_c}")

    if t_c is not None and null_t_c is not None:
        delta = abs(t_c - null_t_c)
        dt = sim.dt * diag_interval
        print(f"  Agreement      : {delta:.3f} ({delta/dt:.1f} diag steps) "
              f"{'✅ CONSISTENT' if delta < 3*dt else '⚠️ CHECK'}")
    elif t_c is None:
        print("  ⚠️ No topological unlinking within t=2.0 — run needs longer steps")
    elif null_t_c is None:
        print("  ⚠️ No spiral null detected — check null finder threshold")

    total_min = (time.time() - t0_wall) / 60
    print(f"\n  Wall time: {total_min:.1f} minutes")
    print(f"  Per-step : {total_min*60/steps:.1f} seconds")
    print(f"  Projected 600-step run: {total_min * 600/steps:.0f} minutes")
    print("=" * 60)

if __name__ == "__main__":
    run_verification()

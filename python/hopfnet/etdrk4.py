"""
Milestone 6: The Kassam-Trefethen ETDRK4 Integrator

Implements the 4th-order Exponential Time Differencing Runge-Kutta scheme
for the stiff incompressible Hall-MHD equations.
"""
import numpy as np
from . import hopfnet_cpp as eng
from .rhs import compute_B_and_J, compute_R_A, compute_R_v

class ETDRK4Integrator:
    def __init__(self, grid, fft, dt, d_i, eta, eta4, nu, nu4):
        self.grid = grid
        self.fft = fft
        self.dt = dt
        self.d_i = d_i
        self.eta = eta
        self.eta4 = eta4

        # 1. Precompute the linear operators via the C++ backend
        # (Evaluated perfectly at every mode via 32-point complex contour,
        #  R=1/dt per Kassam-Trefethen so the contour encloses all eigenvalues)
        c_A = eng.ETDCoeffs(grid, dt, eta,  eta4, 32)
        c_v = eng.ETDCoeffs(grid, dt, nu,   nu4,  32)

        # Cache all coefficient arrays immediately — zero allocation in the hot loop.
        # Naming: E1 = exp(L*dt/2), E2 = exp(L*dt); f1/f2/f3 = KT phi functions.
        self.E1_A = c_A.e1(grid)   # exp(L_A * dt/2)
        self.E2_A = c_A.e2(grid)   # exp(L_A * dt)
        self.f1_A = c_A.f1(grid)
        self.f2_A = c_A.f2(grid)
        self.f3_A = c_A.f3(grid)

        self.E1_v = c_v.e1(grid)   # exp(L_v * dt/2)
        self.E2_v = c_v.e2(grid)   # exp(L_v * dt)
        self.f1_v = c_v.f1(grid)
        self.f2_v = c_v.f2(grid)
        self.f3_v = c_v.f3(grid)

        # 2. Half-step propagator Q = (exp(L*dt/2) - 1) / L
        # Exact identity avoiding 0/0: Q = f1 / (E1 + 1)
        # where E1 = exp(L*dt/2) — the HALF-step exponential (not E2).
        self.Q_A = self.f1_A / (self.E1_A + 1.0)
        self.Q_v = self.f1_v / (self.E1_v + 1.0)

    def _eval_rhs(self, A_hat, v_hat):
        """Evaluates the nonlinear terms for Induction and Momentum."""
        B_hat, J_hat = compute_B_and_J(self.grid, A_hat)
        RA_hat = compute_R_A(self.grid, self.fft, v_hat, B_hat, J_hat, self.eta, self.d_i)
        Rv_hat = compute_R_v(self.grid, self.fft, v_hat, B_hat, J_hat)
        return RA_hat, Rv_hat

    def step(self, A_n, v_n):
        """Advances the state by one timestep dt using ETDRK4."""
        dt = self.dt

        # Stage 1: evaluate nonlinear RHS at t_n
        R_An, R_vn = self._eval_rhs(A_n, v_n)

        # Stage 2: half-step 'a' using Q = (exp(L*dt/2)-1)/L
        # A_a = E1*A_n + dt*Q*N_n  (advances to t + dt/2)
        A_a = tuple(self.E1_A * A_n[i] + dt * self.Q_A * R_An[i] for i in range(3))
        v_a = tuple(self.E1_v * v_n[i] + dt * self.Q_v * R_vn[i] for i in range(3))
        R_Aa, R_va = self._eval_rhs(A_a, v_a)

        # Stage 3: half-step 'b' using updated midpoint RHS
        A_b = tuple(self.E1_A * A_n[i] + dt * self.Q_A * R_Aa[i] for i in range(3))
        v_b = tuple(self.E1_v * v_n[i] + dt * self.Q_v * R_va[i] for i in range(3))
        R_Ab, R_vb = self._eval_rhs(A_b, v_b)

        # Stage 4: full-step 'c' using corrected midpoint
        # A_c = E1*A_a + dt*Q*(2*N_b - N_n)  (full step from the midpoint)
        A_c = tuple(self.E1_A * A_a[i] + dt * self.Q_A * (2*R_Ab[i] - R_An[i]) for i in range(3))
        v_c = tuple(self.E1_v * v_a[i] + dt * self.Q_v * (2*R_vb[i] - R_vn[i]) for i in range(3))
        R_Ac, R_vc = self._eval_rhs(A_c, v_c)

        # Final update: Kassam-Trefethen ETDRK4 (their Eq. 2.5)
        # u_{n+1} = E2*u_n + dt*[f1*N_n + 2*f2*(N_a + N_b) + f3*N_c]
        # f1, f2, f3 are the precomputed phi functions from ETDCoeffs —
        # no separate W_A/W_B/W_C needed; those are exactly f1, 2*f2, f3.
        A_next = tuple(
            self.E2_A * A_n[i] + dt * (
                self.f1_A * R_An[i]
                + 2.0 * self.f2_A * (R_Aa[i] + R_Ab[i])
                + self.f3_A * R_Ac[i]
            ) for i in range(3)
        )
        v_next = tuple(
            self.E2_v * v_n[i] + dt * (
                self.f1_v * R_vn[i]
                + 2.0 * self.f2_v * (R_va[i] + R_vb[i])
                + self.f3_v * R_vc[i]
            ) for i in range(3)
        )

        # Guarantee Coulomb gauge and div(v)=0 explicitly to scrub floating-point drift
        A_next = eng.project_field(self.grid, *A_next)
        v_next = eng.project_field(self.grid, *v_next)

        return A_next, v_next
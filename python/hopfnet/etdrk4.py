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
        # (Evaluated perfectly at every mode via 32-point complex contour)
        self.c_A = eng.precompute_etdrk4(grid, dt, eta, eta4, M=32, R=1.0)
        self.c_v = eng.precompute_etdrk4(grid, dt, nu, nu4, M=32, R=1.0)

        # 2. Derive the half-step coefficient Q analytically to avoid 0/0
        # Identity: (e^(c/2) - 1)/c = (e^c - 1)/c * 1/(e^(c/2) + 1) = f1 / (E2 + 1)
        self.Q_A = self.c_A.f1 / (self.c_A.E2 + 1.0)
        self.Q_v = self.c_v.f1 / (self.c_v.E2 + 1.0)

        # 3. Compute Kassam-Trefethen ETDRK4 final update weights (A, B, C) via contour
        k2 = grid.k2()
        L_A_dt = (-eta * k2 - eta4 * k2**2) * dt
        L_v_dt = (-nu * k2 - nu4 * k2**2) * dt

        self.W_A_A, self.W_B_A, self.W_C_A = self._compute_kt_weights(L_A_dt)
        self.W_A_v, self.W_B_v, self.W_C_v = self._compute_kt_weights(L_v_dt)

    def _compute_kt_weights(self, c, M=32, R=1.0):
        """Computes Kassam-Trefethen RK4 weights A, B, C via contour integral."""
        A_w = np.zeros_like(c, dtype=np.complex128)
        B_w = np.zeros_like(c, dtype=np.complex128)
        C_w = np.zeros_like(c, dtype=np.complex128)
        for j in range(M):
            theta = 2 * np.pi * j / M
            z = c + R * np.exp(1j * theta)
            ez = np.exp(z)
            A_w += (-4 - z + ez * (4 - 3*z + z**2)) / z**3
            B_w += (2 + z + ez * (-2 + z)) / z**3
            C_w += (-4 - 3*z - z**2 + ez * (4 - z)) / z**3
        return A_w / M, B_w / M, C_w / M

    def _eval_rhs(self, A_hat, v_hat):
        """Evaluates the nonlinear terms for Induction and Momentum."""
        B_hat, J_hat = compute_B_and_J(self.grid, A_hat)
        RA_hat = compute_R_A(self.grid, self.fft, v_hat, B_hat, J_hat, self.eta, self.d_i)
        Rv_hat = compute_R_v(self.grid, self.fft, v_hat, B_hat, J_hat)
        return RA_hat, Rv_hat

    def step(self, A_n, v_n):
        """Advances the state by one timestep dt using ETDRK4."""
        dt = self.dt

        # Stage 1: N_n
        R_An, R_vn = self._eval_rhs(A_n, v_n)

        # Stage 2: a
        A_a = tuple(self.c_A.E2 * A_n[i] + dt * self.Q_A * R_An[i] for i in range(3))
        v_a = tuple(self.c_v.E2 * v_n[i] + dt * self.Q_v * R_vn[i] for i in range(3))
        R_Aa, R_va = self._eval_rhs(A_a, v_a)

        # Stage 3: b
        A_b = tuple(self.c_A.E2 * A_n[i] + dt * self.Q_A * R_Aa[i] for i in range(3))
        v_b = tuple(self.c_v.E2 * v_n[i] + dt * self.Q_v * R_va[i] for i in range(3))
        R_Ab, R_vb = self._eval_rhs(A_b, v_b)

        # Stage 4: c
        A_c = tuple(self.c_A.E2 * A_a[i] + dt * self.Q_A * (2*R_Ab[i] - R_An[i]) for i in range(3))
        v_c = tuple(self.c_v.E2 * v_a[i] + dt * self.Q_v * (2*R_vb[i] - R_vn[i]) for i in range(3))
        R_Ac, R_vc = self._eval_rhs(A_c, v_c)

        # Final Update
        A_next = tuple(
            self.c_A.E * A_n[i] + dt * (
                self.W_A_A * R_An[i] + self.W_B_A * (R_Aa[i] + R_Ab[i]) + self.W_C_A * R_Ac[i]
            ) for i in range(3)
        )
        v_next = tuple(
            self.c_v.E * v_n[i] + dt * (
                self.W_A_v * R_vn[i] + self.W_B_v * (R_va[i] + R_vb[i]) + self.W_C_v * R_vc[i]
            ) for i in range(3)
        )

        # Guarantee Coulomb gauge and div(v)=0 explicitly to scrub floating-point drift
        A_next = eng.project_field(self.grid, *A_next)
        v_next = eng.project_field(self.grid, *v_next)

        return A_next, v_next
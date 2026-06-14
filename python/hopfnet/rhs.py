"""
Milestone 4: nonlinear RHS pipeline — Incompressible Hall-MHD.

PHYSICAL REGIME
---------------
We operate strictly in the Incompressible Hall-MHD limit:

    rho = 1  (global constant, normalized to unity)
    div(v) = 0  (enforced spectrally by P(k) on every RHS evaluation)

This filters acoustic (sound) waves, which are irrelevant to magnetic
reconnection and would otherwise severely restrict the ETDRK4 timestep.
The surviving wave modes — Hall whistler waves and Alfvén waves — are
exactly the modes that drive topological collapse and are the subject of
the Neural ODE predictor in Phase 3.

INITIAL CONDITION FOR THE VELOCITY FIELD
-----------------------------------------
The plasma is initialized at rest:  v(t=0) = 0.
The velocity field is driven entirely by the Lorentz force J x B as the
Hopf link attempts Taylor relaxation. This is the physically correct choice:
the only source of momentum in the initial state is the magnetic stress of
the interlocked flux rings.

FUNCTIONS
---------
  - spectral_derivative:  d/dx_j in Fourier space (multiplication by i*k_j)
  - compute_B_and_J:      B = curl(A), J = curl(B)
  - compute_advection:    (v . grad) v, dealiased
  - compute_R_A:          R = v x B - (d_i)(J x B) - eta*J   (Eq. 12),
                           dealiased and Coulomb-projected (Eq. 18).
                           rho=1 absorbed into d_i parameter.
  - compute_R_v:          -(v.grad)v + (J x B), dealiased and
                           pressure-projected with the same P(k) as above,
                           eliminating grad(P) and enforcing div(v)=0.
                           rho=1 makes the Lorentz term dimensionless.

All "_hat" quantities are complex spectral fields of shape (N, N, N/2+1).
All "_real" quantities are real (N, N, N) fields.
Vector fields are length-3 tuples of arrays (x, y, z components).

The stiff linear operators
    L_A = -eta*k^2 - eta4*k^4   (magnetic hyper-resistivity)
    L_v = -nu*k^2  - nu4*k^4    (viscous hyper-diffusion)
are handled by the ETDRK4 coefficient sets in Milestone 5 and are NOT
part of R_A / R_v here.
"""
import numpy as np

from . import hopfnet_cpp as eng


def _k_arrays(grid):
    kx, ky, kz = grid.kx(), grid.ky(), grid.kz()
    KX = kx[:, None, None]
    KY = ky[None, :, None]
    KZ = kz[None, None, :]
    return KX, KY, KZ


def spectral_derivative(grid, field_hat, axis):
    """Return d(field)/dx_axis in Fourier space: i*k_axis * field_hat."""
    K = _k_arrays(grid)[axis]
    return 1j * K * field_hat


def to_real(fft, hats):
    return tuple(fft.inverse(h) for h in hats)


def to_hat(fft, reals):
    return tuple(fft.forward(r) for r in reals)


def apply_dealias(grid, hats):
    mask = grid.dealias_mask()
    return tuple(h * mask for h in hats)


def cross_real(a, b):
    """Elementwise cross product of two real vector fields (3-tuples)."""
    ax, ay, az = a
    bx, by, bz = b
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    return cx, cy, cz


def compute_B_and_J(grid, A_hat):
    """B_hat = curl(A_hat), J_hat = curl(B_hat)."""
    B_hat = eng.spectral_curl(grid, *A_hat)
    J_hat = eng.spectral_curl(grid, *B_hat)
    return B_hat, J_hat


def compute_advection(grid, fft, v_hat):
    """(v . grad) v, computed pseudo-spectrally and dealiased."""
    v_real = to_real(fft, v_hat)

    adv_real = [np.zeros_like(v_real[0]) for _ in range(3)]
    for i in range(3):
        for j in range(3):
            dvi_dxj_hat = spectral_derivative(grid, v_hat[i], j)
            dvi_dxj_real = fft.inverse(dvi_dxj_hat)
            adv_real[i] = adv_real[i] + v_real[j] * dvi_dxj_real

    adv_hat = to_hat(fft, tuple(adv_real))
    adv_hat = apply_dealias(grid, adv_hat)
    return adv_hat


def compute_R_A(grid, fft, v_hat, B_hat, J_hat, eta, d_i):
    """
    Hall-MHD induction RHS (Eq. 12, dealiased and Coulomb-projected):

        R_A = v x B - d_i*(J x B) - eta*J

    rho=1 is absorbed: d_i here is already d_i/rho with rho=1.
    """
    v_real = to_real(fft, v_hat)
    B_real = to_real(fft, B_hat)
    J_real = to_real(fft, J_hat)

    vxB = cross_real(v_real, B_real)
    JxB = cross_real(J_real, B_real)

    R_real = tuple(vxB[c] - d_i * JxB[c] - eta * J_real[c] for c in range(3))

    R_hat = to_hat(fft, R_real)
    R_hat = apply_dealias(grid, R_hat)
    R_hat = eng.project_field(grid, *R_hat)
    return R_hat


def compute_R_v(grid, fft, v_hat, B_hat, J_hat):
    """
    Incompressible NS nonlinear forcing (rho=1), dealiased and
    pressure-projected via the same P(k) that enforces div(v)=0:

        R_v = -(v.grad)v + (J x B)
    """
    adv_hat = compute_advection(grid, fft, v_hat)

    B_real = to_real(fft, B_hat)
    J_real = to_real(fft, J_hat)
    JxB_real = cross_real(J_real, B_real)
    JxB_hat = to_hat(fft, JxB_real)
    JxB_hat = apply_dealias(grid, JxB_hat)

    R_hat = tuple(-adv_hat[c] + JxB_hat[c] for c in range(3))
    R_hat = eng.project_field(grid, *R_hat)
    return R_hat

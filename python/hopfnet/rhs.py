"""
Milestone 4: nonlinear RHS pipeline.

Implements, in terms of the C++ primitives from Milestones 1-3
(FFT3D, SpectralGrid, project_field, spectral_curl):

  - spectral_derivative:  d/dx_j in Fourier space (multiplication by i*k_j)
  - compute_B_and_J:      B = curl(A), J = curl(B)
  - compute_advection:    (v . grad) v, dealiased
  - compute_R_A:          R = v x B - (d_i/rho)(J x B) - eta*J   (Eq. 12),
                           dealiased and Coulomb-projected (Eq. 18)
  - compute_R_v:          incompressible Navier-Stokes nonlinear forcing
                           -(v.grad)v + (1/rho)(J x B), dealiased and
                           pressure-projected (same P(k) tensor as Eq. 18,
                           applied to velocity rather than vector potential)

All "_hat" quantities are complex spectral fields of shape (N, N, N/2+1);
all "_real" quantities are real (N, N, N) fields. Vector fields are
represented as length-3 tuples of arrays (x, y, z components).

The linear (stiff) operators L_A = -eta*k^2 - nu4_A*k^4 and
L_v = -nu*k^2 - nu4_v*k^4 are handled separately by the ETDRK4
coefficients (Milestone 5) and are NOT part of R_A / R_v here.
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


def compute_R_A(grid, fft, v_hat, B_hat, J_hat, eta, d_i, rho=1.0):
    """
    Hall-MHD induction RHS (Eq. 12, dealiased and projected via Eq. 18):

        R = v x B - (d_i/rho)(J x B) - eta*J
    """
    v_real = to_real(fft, v_hat)
    B_real = to_real(fft, B_hat)
    J_real = to_real(fft, J_hat)

    vxB = cross_real(v_real, B_real)
    JxB = cross_real(J_real, B_real)

    R_real = tuple(vxB[c] - (d_i / rho) * JxB[c] - eta * J_real[c] for c in range(3))

    R_hat = to_hat(fft, R_real)
    R_hat = apply_dealias(grid, R_hat)
    R_hat = eng.project_field(grid, *R_hat)
    return R_hat


def compute_R_v(grid, fft, v_hat, B_hat, J_hat, rho=1.0):
    """
    Incompressible Navier-Stokes nonlinear forcing, dealiased and
    pressure-projected with the same P(k) tensor used for the Coulomb
    gauge (here enforcing div(v) = 0):

        R_v = -(v . grad) v + (1/rho)(J x B)
    """
    adv_hat = compute_advection(grid, fft, v_hat)

    B_real = to_real(fft, B_hat)
    J_real = to_real(fft, J_hat)
    JxB_real = cross_real(J_real, B_real)
    JxB_hat = to_hat(fft, JxB_real)
    JxB_hat = apply_dealias(grid, JxB_hat)

    R_hat = tuple(-adv_hat[c] + (1.0 / rho) * JxB_hat[c] for c in range(3))
    R_hat = eng.project_field(grid, *R_hat)
    return R_hat

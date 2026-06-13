#pragma once
#include <complex>
#include <vector>
#include "spectral_grid.hpp"

using cplx = std::complex<double>;

// Apply the Coulomb-gauge orthogonal projection tensor
//     P(k) = I - (k ⊗ k) / |k|^2
// to a vector field given in Fourier space, in place.
//
// At k=0, P(0) ≡ I by definition (Sec. 2.3 of mathematical_foundation.tex):
// the k=0 mode represents the spatially-uniform mean field, which trivially
// satisfies ∇·A = 0 and carries no longitudinal component to remove.
inline void project_field(const SpectralGrid& grid,
                           std::vector<cplx>& Ax,
                           std::vector<cplx>& Ay,
                           std::vector<cplx>& Az) {
    for (int i = 0; i < grid.N; ++i) {
        for (int j = 0; j < grid.N; ++j) {
            for (int k = 0; k < grid.Nz_half; ++k) {
                const size_t idx = grid.index(i, j, k);
                const double k2 = grid.k2[idx];
                if (k2 < 1e-14) continue; // k=0: P(0) = I, no-op

                const double kx_ = grid.kx[i], ky_ = grid.ky[j], kz_ = grid.kz[k];
                const cplx kdotA = kx_ * Ax[idx] + ky_ * Ay[idx] + kz_ * Az[idx];
                const cplx factor = kdotA / k2;

                Ax[idx] -= kx_ * factor;
                Ay[idx] -= ky_ * factor;
                Az[idx] -= kz_ * factor;
            }
        }
    }
}

// Spectral curl: B_hat(k) = i k x A_hat(k)
//   Bx = i (ky*Az - kz*Ay)
//   By = i (kz*Ax - kx*Az)
//   Bz = i (kx*Ay - ky*Ax)
inline void spectral_curl(const SpectralGrid& grid,
                           const std::vector<cplx>& Ax,
                           const std::vector<cplx>& Ay,
                           const std::vector<cplx>& Az,
                           std::vector<cplx>& Bx,
                           std::vector<cplx>& By,
                           std::vector<cplx>& Bz) {
    auto times_i = [](cplx c) { return cplx(-c.imag(), c.real()); };

    const size_t total = (size_t)grid.N * grid.N * grid.Nz_half;
    Bx.resize(total);
    By.resize(total);
    Bz.resize(total);

    for (int i = 0; i < grid.N; ++i) {
        for (int j = 0; j < grid.N; ++j) {
            for (int k = 0; k < grid.Nz_half; ++k) {
                const size_t idx = grid.index(i, j, k);
                const double kx_ = grid.kx[i], ky_ = grid.ky[j], kz_ = grid.kz[k];

                Bx[idx] = times_i(ky_ * Az[idx] - kz_ * Ay[idx]);
                By[idx] = times_i(kz_ * Ax[idx] - kx_ * Az[idx]);
                Bz[idx] = times_i(kx_ * Ay[idx] - ky_ * Ax[idx]);
            }
        }
    }
}

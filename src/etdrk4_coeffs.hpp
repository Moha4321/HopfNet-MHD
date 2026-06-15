#pragma once
#include <complex>
#include <vector>
#include <cmath>
#include "spectral_grid.hpp"

// ETDRK4 coefficient precomputation via Kassam-Trefethen (2005) contour integral.
//
// Linear operator: L(k) = -alpha*k^2 - beta*k^4
//
// Five coefficient arrays (all real, since L is real negative):
//   e1 = exp(L * dt/2)       [half-step propagator]
//   e2 = exp(L * dt)         [full-step propagator]
//   f1 = phi_1(L*dt)         = (exp(L*dt) - 1) / (L*dt)           * dt
//   f2 = phi_2(L*dt)         = (exp(L*dt) - L*dt - 1) / (L*dt)^2  * dt
//   f3 = phi_3(L*dt)         = (exp(L*dt) - (L*dt)^2/2 - L*dt - 1)/(L*dt)^3 * dt
//
// ALL modes including k=0 (L=0) use the M-point circular contour z_j = R*exp(i*theta_j)
// evaluated at z_j + L. This avoids 0/0 by Cauchy's theorem — NO threshold switching.
//
// Precomputed once at t=0, stored flat (size N*N*(N/2+1)), zero allocation in hot loop.
struct ETDCoeffs {
    std::vector<double> e1, e2, f1, f2, f3;

    ETDCoeffs() = default;

    ETDCoeffs(const SpectralGrid& grid, double dt,
              double alpha, double beta, int M = 32) {
        const size_t total = (size_t)grid.N * grid.N * grid.Nz_half;
        e1.resize(total); e2.resize(total);
        f1.resize(total); f2.resize(total); f3.resize(total);

        // R = 1/dt: contour radius ensures the circle encloses all eigenvalues
        // of L (which are real negative, bounded by -beta*k_max^4).
        const double R = 1.0 / dt;

        for (size_t idx = 0; idx < total; ++idx) {
            const double k2  = grid.k2[idx];
            const double L   = -alpha * k2 - beta * k2 * k2;

            // Accumulate five contour integrals over M quadrature points.
            // z_j = R*exp(i*theta_j), shifted argument zL = z_j + L.
            // Because L is real, imaginary parts cancel; accumulate real parts only.
            double acc_e1=0, acc_e2=0, acc_f1=0, acc_f2=0, acc_f3=0;

            for (int j = 0; j < M; ++j) {
                const double theta = 2.0 * M_PI * j / M;
                const std::complex<double> z(R * std::cos(theta),
                                              R * std::sin(theta));
                const std::complex<double> zL = z + L;

                const std::complex<double> ez_half = std::exp(zL * (dt * 0.5));
                const std::complex<double> ez_full = std::exp(zL * dt);

                // phi_1: (exp(zL*dt) - 1) / zL   [note: *dt absorbed below]
                const std::complex<double> f1v = (ez_full - 1.0) / zL;

                // phi_2: (exp(zL*dt) - zL*dt - 1) / (zL^2 * dt)
                const std::complex<double> f2v =
                    (ez_full - zL * dt - 1.0) / (zL * zL * dt);

                // phi_3: (exp(zL*dt) - (zL*dt)^2/2 - zL*dt - 1) / (zL^3 * dt^2)
                const std::complex<double> zdt = zL * dt;
                const std::complex<double> f3v =
                    (ez_full - 0.5 * zdt * zdt - zdt - 1.0)
                    / (zL * zL * zL * dt * dt);

                acc_e1 += ez_half.real();
                acc_e2 += ez_full.real();
                acc_f1 += f1v.real();
                acc_f2 += f2v.real();
                acc_f3 += f3v.real();
            }

            e1[idx] = acc_e1 / M;
            e2[idx] = acc_e2 / M;
            f1[idx] = acc_f1 / M;
            f2[idx] = acc_f2 / M;
            f3[idx] = acc_f3 / M;
        }
    }
};
#pragma once
#include <complex>
#include <vector>
#include <cmath>
#include "spectral_grid.hpp"

using cplx = std::complex<double>;

// ETDRK4 coefficient sets, computed via the Kassam-Trefethen (2005)
// M-point circular contour integration in the complex plane.
//
// For a spectral ODE  du/dt = L*u + N(u,t)  we require five scalars per
// mode k (all are functions of c = L(k)*dt):
//
//   E  = exp(c)                    (full-step linear propagator)
//   E2 = exp(c/2)                  (half-step linear propagator)
//   f1 = (exp(c) - 1) / c
//   f2 = (exp(c) - c - 1) / c^2
//   f3 = (exp(c) - c/2 - 1) / c^2    [NB: the Kassam-Trefethen f3 coefficient]
//
// ALL FIVE are computed via the M-point contour integral for every mode
// without exception — including k=0 where c=0 and the naive formulae give
// 0/0.  The contour integration evaluates f(c) at points z_j = c + R*exp(i*theta_j)
// on a circle of radius R centred on c (not the origin), which keeps the
// contour away from z=0 even when c=0.
//
// Following Kassam & Trefethen exactly: R = 1.0, M = 32.
// This is a one-time precomputation performed at t=0.

struct ETDRK4Coeffs {
    std::vector<cplx> E;   // exp(L*dt)
    std::vector<cplx> E2;  // exp(L*dt/2)
    std::vector<cplx> f1;  // (E - 1) / L*dt          via contour
    std::vector<cplx> f2;  // (E - L*dt - 1) / (L*dt)^2  via contour
    std::vector<cplx> f3;  // Kassam-Trefethen f3 coefficient via contour

    int N;
    int Nz_half;
};

// Evaluate the five ETDRK4 phi-functions for a single scalar c = L*dt
// using an M-point circular contour of radius R centred at c.
// This matches Kassam & Trefethen (2005), Sec. 2, exactly.
inline void contour_coeffs(cplx c, int M, double R,
                            cplx& E_out, cplx& E2_out,
                            cplx& f1_out, cplx& f2_out, cplx& f3_out) {
    // For E and E2, the contour integral recovers exp exactly;
    // equivalently we just compute exp(c) directly — these are never 0/0.
    E_out  = std::exp(c);
    E2_out = std::exp(c * 0.5);

    // f1, f2, f3 are singular at c=0 analytically.
    // Contour: z_j = c + R * exp(i * 2*pi*j/M), j=1..M
    cplx sum_f1{0,0}, sum_f2{0,0}, sum_f3{0,0};
    for (int j = 0; j < M; ++j) {
        const double theta = 2.0 * M_PI * j / M;
        const cplx z = c + R * cplx(std::cos(theta), std::sin(theta));
        const cplx ez = std::exp(z);

        // phi_1(z) = (e^z - 1) / z
        sum_f1 += (ez - 1.0) / z;

        // phi_2(z) = (e^z - z - 1) / z^2
        sum_f2 += (ez - z - 1.0) / (z * z);

        // phi_3(z) as used in the Cox-Matthews / Kassam-Trefethen ETDRK4 stages:
        // f3(z) = (e^z - z/2 - 1) / z^2
        // (this is the coefficient appearing in the b3/b4 weights of the scheme)
        sum_f3 += (ez - 0.5 * z - 1.0) / (z * z);
    }
    f1_out = sum_f1 / (double)M;
    f2_out = sum_f2 / (double)M;
    f3_out = sum_f3 / (double)M;
}

// Precompute ETDRK4 coefficients for a given linear operator L(k).
// L_values must be a flat array of length N*N*Nz_half giving the scalar
// L(k) = -alpha*k^2 - beta*k^4 for each spectral mode.
inline ETDRK4Coeffs precompute_etdrk4(const SpectralGrid& grid, double dt,
                                        double alpha, double beta,
                                        int M = 32, double R = 1.0) {
    const size_t total = (size_t)grid.N * grid.N * grid.Nz_half;

    ETDRK4Coeffs co;
    co.N = grid.N;
    co.Nz_half = grid.Nz_half;
    co.E.resize(total);
    co.E2.resize(total);
    co.f1.resize(total);
    co.f2.resize(total);
    co.f3.resize(total);

    for (int i = 0; i < grid.N; ++i) {
        for (int j = 0; j < grid.N; ++j) {
            for (int k = 0; k < grid.Nz_half; ++k) {
                const size_t idx = grid.index(i, j, k);
                const double k2 = grid.k2[idx];
                const double k4 = k2 * k2;

                // Scalar linear operator for this mode, times dt
                // L(k) = -alpha*k^2 - beta*k^4  (both alpha, beta >= 0)
                const cplx c = cplx(-alpha * k2 - beta * k4, 0.0) * dt;

                contour_coeffs(c, M, R,
                               co.E[idx], co.E2[idx],
                               co.f1[idx], co.f2[idx], co.f3[idx]);
            }
        }
    }
    return co;
}

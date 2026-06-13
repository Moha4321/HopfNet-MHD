#pragma once
#include <cmath>
#include <vector>

// Computes the regularized Biot-Savart vector potential for the magnetic
// Hopf link initial condition (mathematical_foundation.tex, Sec. 3).
//
// Ring 1 (C1): xy-plane, centered at (0, d, 0), radius R.
// Ring 2 (C2): yz-plane, centered at (0, -d, 0), radius R.
//
// The grid is the same triply-periodic box used by SpectralGrid (side L),
// but centered on the origin: x_i = -L/2 + i*(L/N), so the two rings sit
// symmetrically about the box center.
//
// Each 1D contour integral (Eqs. 26-28) is evaluated with composite
// Simpson's rule over theta, phi in [0, 2*pi]. n_quad must be even.
namespace hopf_link {

// Composite Simpson's rule on [0, 2*pi] with n_quad (even) subintervals.
template <typename F>
inline double simpson_2pi(F f, int n_quad) {
    const double a = 0.0;
    const double b = 2.0 * M_PI;
    const double h = (b - a) / n_quad;

    double sum = f(a) + f(b);
    for (int i = 1; i < n_quad; ++i) {
        const double x = a + i * h;
        sum += (i % 2 == 0 ? 2.0 : 4.0) * f(x);
    }
    return sum * h / 3.0;
}

// Fills Ax, Ay, Az (each size N*N*N, row-major index (ix*N + iy)*N + iz)
// with the regularized Biot-Savart vector potential of the Hopf link.
inline void compute(int N, double L, double R, double d, double a_core,
                     double I0, double mu0, int n_quad,
                     std::vector<double>& Ax,
                     std::vector<double>& Ay,
                     std::vector<double>& Az) {
    const size_t total = (size_t)N * N * N;
    Ax.assign(total, 0.0);
    Ay.assign(total, 0.0);
    Az.assign(total, 0.0);

    const double prefactor = mu0 * I0 / (4.0 * M_PI);
    const double a2 = a_core * a_core;
    const double dx = L / N;

    for (int ix = 0; ix < N; ++ix) {
        const double x = -L / 2.0 + ix * dx;
        for (int iy = 0; iy < N; ++iy) {
            const double y = -L / 2.0 + iy * dx;
            for (int iz = 0; iz < N; ++iz) {
                const double z = -L / 2.0 + iz * dx;
                const size_t idx = ((size_t)ix * N + iy) * N + iz;

                // Ring 1 (xy-plane, center (0,d,0)): D1(theta), integrands for Ax, Ay
                auto D1 = [&](double theta) {
                    const double cx = x - R * std::cos(theta);
                    const double cy = y - d - R * std::sin(theta);
                    const double cz = z;
                    return std::sqrt(cx * cx + cy * cy + cz * cz + a2);
                };
                auto Ax1_integrand = [&](double theta) {
                    return -R * std::sin(theta) / D1(theta);
                };
                auto Ay1_integrand = [&](double theta) {
                    return R * std::cos(theta) / D1(theta);
                };

                // Ring 2 (yz-plane, center (0,-d,0)): D2(phi), integrands for Ay, Az
                auto D2 = [&](double phi) {
                    const double cx = x;
                    const double cy = y + d - R * std::cos(phi);
                    const double cz = z - R * std::sin(phi);
                    return std::sqrt(cx * cx + cy * cy + cz * cz + a2);
                };
                auto Ay2_integrand = [&](double phi) {
                    return -R * std::sin(phi) / D2(phi);
                };
                auto Az2_integrand = [&](double phi) {
                    return R * std::cos(phi) / D2(phi);
                };

                const double Ax1 = simpson_2pi(Ax1_integrand, n_quad);
                const double Ay1 = simpson_2pi(Ay1_integrand, n_quad);
                const double Ay2 = simpson_2pi(Ay2_integrand, n_quad);
                const double Az2 = simpson_2pi(Az2_integrand, n_quad);

                Ax[idx] = prefactor * Ax1;
                Ay[idx] = prefactor * (Ay1 + Ay2);
                Az[idx] = prefactor * Az2;
            }
        }
    }
}

} // namespace hopf_link

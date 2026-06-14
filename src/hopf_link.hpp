#pragma once
#include <cmath>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// Computes the regularized Biot-Savart vector potential for the magnetic
// Hopf link initial condition (mathematical_foundation.tex, Sec. 3).
//
// The outer loop over grid points is embarrassingly parallel and is
// parallelized with OpenMP.  On the M4's performance cores this reduces
// the 128^3 initialization from O(minutes) single-threaded to a few seconds.
//
// Enable with: target_compile_options(hopfnet_cpp PRIVATE -fopenmp)
//              target_link_libraries(hopfnet_cpp PRIVATE ... OpenMP::OpenMP_CXX)
namespace hopf_link {

template <typename F>
inline double simpson_2pi(F f, int n_quad) {
    const double h = 2.0 * M_PI / n_quad;
    double sum = f(0.0) + f(2.0 * M_PI);
    for (int i = 1; i < n_quad; ++i) {
        const double x = i * h;
        sum += (i % 2 == 0 ? 2.0 : 4.0) * f(x);
    }
    return sum * h / 3.0;
}

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

#pragma omp parallel for schedule(static) collapse(2)
    for (int ix = 0; ix < N; ++ix) {
        for (int iy = 0; iy < N; ++iy) {
            const double x = -L / 2.0 + ix * dx;
            const double y = -L / 2.0 + iy * dx;
            for (int iz = 0; iz < N; ++iz) {
                const double z = -L / 2.0 + iz * dx;
                const size_t idx = ((size_t)ix * N + iy) * N + iz;

                // Ring 1 (xy-plane, center (0,d,0))
                auto D1 = [&](double theta) {
                    double cx = x - R * std::cos(theta);
                    double cy = y - d - R * std::sin(theta);
                    return std::sqrt(cx*cx + cy*cy + z*z + a2);
                };
                auto Ax1_int = [&](double t){ return -R*std::sin(t)/D1(t); };
                auto Ay1_int = [&](double t){ return  R*std::cos(t)/D1(t); };

                // Ring 2 (yz-plane, center (0,-d,0))
                auto D2 = [&](double phi) {
                    double cy = y + d - R * std::cos(phi);
                    double cz = z - R * std::sin(phi);
                    return std::sqrt(x*x + cy*cy + cz*cz + a2);
                };
                auto Ay2_int = [&](double p){ return -R*std::sin(p)/D2(p); };
                auto Az2_int = [&](double p){ return  R*std::cos(p)/D2(p); };

                Ax[idx] = prefactor * simpson_2pi(Ax1_int, n_quad);
                Ay[idx] = prefactor * (simpson_2pi(Ay1_int, n_quad)
                                     + simpson_2pi(Ay2_int, n_quad));
                Az[idx] = prefactor * simpson_2pi(Az2_int, n_quad);
            }
        }
    }
}

} // namespace hopf_link

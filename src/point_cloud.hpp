#pragma once
#include <vector>
#include <cmath>
#include <complex>
#include "spectral_grid.hpp"
#include "fft3d.hpp"

namespace point_cloud {

inline std::vector<double> extract(const SpectralGrid& grid, FFT3D& fft, 
                                   const std::complex<double>* jx_hat, 
                                   const std::complex<double>* jy_hat, 
                                   const std::complex<double>* jz_hat, 
                                   double threshold = 0.6) {
    const size_t real_total = (size_t)grid.N * grid.N * grid.N;
    const size_t cplx_total = (size_t)grid.N * grid.N * grid.Nz_half;

    // 1. Transform J_hat to real space
    std::vector<double> jx(real_total), jy(real_total), jz(real_total);
    fft.inverse(jx_hat, jx.data());
    fft.inverse(jy_hat, jy.data());
    fft.inverse(jz_hat, jz.data());

    // 2. Compute f = |J| and find max
    std::vector<double> f(real_total);
    double max_f = 0.0;
    for(size_t i = 0; i < real_total; ++i) {
        f[i] = std::sqrt(jx[i]*jx[i] + jy[i]*jy[i] + jz[i]*jz[i]);
        if(f[i] > max_f) max_f = f[i];
    }

    // 3. Apply threshold mask
    double thresh_val = threshold * max_f;
    std::vector<size_t> active_idx;
    for(size_t i = 0; i < real_total; ++i) {
        if(f[i] >= thresh_val) active_idx.push_back(i);
    }

    // If no points meet the threshold, return empty
    if(active_idx.empty()) return {};

    // 4. Forward FFT of f to compute spectral derivatives
    std::vector<std::complex<double>> f_hat(cplx_total);
    fft.forward(f.data(), f_hat.data());

    // Helper lambda to compute spectral derivatives (d1, d2)
    // -1 means no derivative. e.g., (0, -1) is d/dx. (0, 1) is d^2/dxdy
    auto compute_deriv = [&](int d1, int d2, std::vector<double>& out) {
        std::vector<std::complex<double>> temp_hat(cplx_total);
        for(int i=0; i<grid.N; ++i) {
            for(int j=0; j<grid.N; ++j) {
                for(int k=0; k<grid.Nz_half; ++k) {
                    size_t idx = grid.index(i, j, k);
                    double k1 = (d1==0) ? grid.kx[i] : (d1==1) ? grid.ky[j] : (d1==2) ? grid.kz[k] : 0;
                    double k2 = (d2==0) ? grid.kx[i] : (d2==1) ? grid.ky[j] : (d2==2) ? grid.kz[k] : 0;
                    
                    if (d2 == -1) { 
                        // First derivative: multiply by (i * k1)
                        temp_hat[idx] = std::complex<double>(-f_hat[idx].imag() * k1, f_hat[idx].real() * k1);
                    } else { 
                        // Second derivative: multiply by (-k1 * k2)
                        temp_hat[idx] = -k1 * k2 * f_hat[idx];
                    }
                }
            }
        }
        fft.inverse(temp_hat.data(), out.data());
    };

    // 5. Compute the 3 Gradients and 6 unique Hessians
    std::vector<double> gx(real_total), gy(real_total), gz(real_total);
    compute_deriv(0, -1, gx); compute_deriv(1, -1, gy); compute_deriv(2, -1, gz);

    std::vector<double> hxx(real_total), hyy(real_total), hzz(real_total);
    std::vector<double> hxy(real_total), hxz(real_total), hyz(real_total);
    compute_deriv(0, 0, hxx); compute_deriv(1, 1, hyy); compute_deriv(2, 2, hzz);
    compute_deriv(0, 1, hxy); compute_deriv(0, 2, hxz); compute_deriv(1, 2, hyz);

    // 6. Assemble the Shape Operator for active points
    std::vector<double> result;
    result.reserve(active_idx.size() * 9);

    for(size_t idx : active_idx) {
        // Recover 3D grid coordinates from flat real-space index
        int rem = idx;
        int r_k = rem % grid.N; rem /= grid.N;
        int r_j = rem % grid.N; rem /= grid.N;
        int r_i = rem;
        
        double x = -grid.L/2.0 + r_i * (grid.L/grid.N);
        double y = -grid.L/2.0 + r_j * (grid.L/grid.N);
        double z = -grid.L/2.0 + r_k * (grid.L/grid.N);

        double grad_mag = std::sqrt(gx[idx]*gx[idx] + gy[idx]*gy[idx] + gz[idx]*gz[idx]);
        if(grad_mag < 1e-12) grad_mag = 1e-12; // Prevent divide-by-zero at flat regions
        
        double nx = gx[idx] / grad_mag;
        double ny = gy[idx] / grad_mag;
        double nz = gz[idx] / grad_mag;

        double H[3][3] = {
            {hxx[idx], hxy[idx], hxz[idx]},
            {hxy[idx], hyy[idx], hyz[idx]},
            {hxz[idx], hyz[idx], hzz[idx]}
        };

        // Projection matrix P = I - n n^T
        double P[3][3] = {
            {1.0 - nx*nx, -nx*ny, -nx*nz},
            {-ny*nx, 1.0 - ny*ny, -ny*nz},
            {-nz*nx, -nz*ny, 1.0 - nz*nz}
        };

        // Shape Operator kappa = (1 / |grad f|) * P * H * P
        // Explicitly projecting on both sides ensures perfect mathematical symmetry in 3D
        double PH[3][3] = {0};
        for(int a=0; a<3; ++a)
            for(int b=0; b<3; ++b)
                for(int c=0; c<3; ++c)
                    PH[a][b] += P[a][c] * H[c][b];

        double kappa[3][3] = {0};
        for(int a=0; a<3; ++a)
            for(int b=0; b<3; ++b)
                for(int c=0; c<3; ++c)
                    kappa[a][b] += PH[a][c] * P[c][b] / grad_mag;

        // Push exactly 9 features: (x, y, z, k11, k12, k13, k22, k23, k33)
        result.push_back(x);
        result.push_back(y);
        result.push_back(z);
        result.push_back(kappa[0][0]);
        result.push_back(kappa[0][1]);
        result.push_back(kappa[0][2]);
        result.push_back(kappa[1][1]);
        result.push_back(kappa[1][2]);
        result.push_back(kappa[2][2]);
    }

    return result;
}

} // namespace point_cloud
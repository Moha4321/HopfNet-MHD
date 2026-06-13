#pragma once
#include <vector>
#include <cmath>
#include <cstdlib>

// SpectralGrid precomputes the wavevector components, |k|^2, and the
// Orszag 2/3 dealiasing mask for a real-to-complex (rFFT) layout of an
// N x N x N triply-periodic domain of side length L.
//
// rFFT layout: real array is N x N x N, complex array is N x N x (N/2+1),
// with the last axis (z) holding only non-negative wavenumbers
// 0 .. N/2 (the Hermitian-redundant negative-z modes are omitted).
struct SpectralGrid {
    int N;
    double L;
    int Nz_half;

    std::vector<double> kx, ky, kz; // 1D wavenumber arrays (sizes N, N, Nz_half)
    std::vector<double> k2;         // flattened 3D |k|^2, size N*N*Nz_half
    std::vector<unsigned char> dealias_mask; // same size, 1 = keep, 0 = zero

    SpectralGrid(int N_, double L_) : N(N_), L(L_), Nz_half(N_ / 2 + 1) {
        const double dk = 2.0 * M_PI / L;

        kx.resize(N);
        ky.resize(N);
        kz.resize(Nz_half);

        for (int i = 0; i < N; ++i) {
            int n = (i <= N / 2) ? i : i - N; // standard FFT wavenumber ordering
            kx[i] = dk * n;
            ky[i] = dk * n;
        }
        for (int k = 0; k < Nz_half; ++k) {
            kz[k] = dk * k;
        }

        const size_t total = (size_t)N * N * Nz_half;
        k2.resize(total);
        dealias_mask.resize(total);

        // Orszag 2/3 rule: keep a mode only if its integer wavenumber along
        // every axis satisfies |n| <= floor((2/3) * (N/2)).
        const int cutoff = (int)std::floor((2.0 / 3.0) * (N / 2));

        for (int i = 0; i < N; ++i) {
            int nx = (i <= N / 2) ? i : i - N;
            for (int j = 0; j < N; ++j) {
                int ny = (j <= N / 2) ? j : j - N;
                for (int k = 0; k < Nz_half; ++k) {
                    int nz = k;
                    const size_t idx = index(i, j, k);
                    k2[idx] = kx[i] * kx[i] + ky[j] * ky[j] + kz[k] * kz[k];

                    const bool keep = (std::abs(nx) <= cutoff) &&
                                      (std::abs(ny) <= cutoff) &&
                                      (std::abs(nz) <= cutoff);
                    dealias_mask[idx] = keep ? 1 : 0;
                }
            }
        }
    }

    inline size_t index(int i, int j, int k) const {
        return ((size_t)i * N + j) * Nz_half + k;
    }
};

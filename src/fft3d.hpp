#pragma once
#include <fftw3.h>
#include <algorithm>
#include <complex>
#include <cstring>
#include <vector>

// FFT3D wraps a single N x N x N real-to-complex (rFFT) plan pair.
// forward():  real field (N^3)              -> complex spectrum (N x N x N/2+1)
// inverse():  complex spectrum (N x N x N/2+1) -> real field (N^3), normalized by 1/N^3
//
// FFTW's c2r transform is unnormalized and DESTROYS its complex input buffer,
// so inverse() always copies into the internal buffer first.
class FFT3D {
public:
    explicit FFT3D(int N) : N_(N), Nz_half_(N / 2 + 1) {
        const size_t real_size = (size_t)N_ * N_ * N_;
        const size_t cplx_size = (size_t)N_ * N_ * Nz_half_;

        real_buf_ = (double*)fftw_malloc(sizeof(double) * real_size);
        cplx_buf_ = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * cplx_size);

        forward_plan_ = fftw_plan_dft_r2c_3d(N_, N_, N_, real_buf_, cplx_buf_, FFTW_MEASURE);
        inverse_plan_ = fftw_plan_dft_c2r_3d(N_, N_, N_, cplx_buf_, real_buf_, FFTW_MEASURE);
    }

    ~FFT3D() {
        fftw_destroy_plan(forward_plan_);
        fftw_destroy_plan(inverse_plan_);
        fftw_free(real_buf_);
        fftw_free(cplx_buf_);
    }

    FFT3D(const FFT3D&) = delete;
    FFT3D& operator=(const FFT3D&) = delete;

    void forward(const double* in, std::complex<double>* out) {
        const size_t real_size = (size_t)N_ * N_ * N_;
        std::copy(in, in + real_size, real_buf_);
        fftw_execute(forward_plan_);
        const size_t cplx_size = (size_t)N_ * N_ * Nz_half_;
        for (size_t i = 0; i < cplx_size; ++i)
            out[i] = std::complex<double>(cplx_buf_[i][0], cplx_buf_[i][1]);
    }

    void inverse(const std::complex<double>* in, double* out) {
        const size_t cplx_size = (size_t)N_ * N_ * Nz_half_;
        for (size_t i = 0; i < cplx_size; ++i) {
            cplx_buf_[i][0] = in[i].real();
            cplx_buf_[i][1] = in[i].imag();
        }
        fftw_execute(inverse_plan_);
        const size_t real_size = (size_t)N_ * N_ * N_;
        const double norm = 1.0 / (double)real_size;
        for (size_t i = 0; i < real_size; ++i)
            out[i] = real_buf_[i] * norm;
    }

    int N() const { return N_; }
    int Nz_half() const { return Nz_half_; }

private:
    int N_, Nz_half_;
    double* real_buf_;
    fftw_complex* cplx_buf_;
    fftw_plan forward_plan_, inverse_plan_;
};

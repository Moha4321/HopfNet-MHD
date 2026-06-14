#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/complex.h>
#include <Eigen/Dense>
#include <complex>
#include <cstring>
#include <vector>

#include "spectral_grid.hpp"
#include "fft3d.hpp"
#include "projection.hpp"
#include "hopf_link.hpp"
#include "etdrk4_coeffs.hpp"

namespace py = pybind11;
using carray = py::array_t<cplx, py::array::c_style | py::array::forcecast>;

// ---- Legacy bridge test ----
Eigen::MatrixXd double_matrix(Eigen::Ref<const Eigen::MatrixXd> input) {
    return input * 2.0;
}

PYBIND11_MODULE(hopfnet_cpp, m) {
    m.doc() = "C++ Physics Engine Backend for HopfNet";
    m.def("double_matrix", &double_matrix, "A function that doubles an Eigen matrix");

    // ---- Milestone 1: SpectralGrid ----
    py::class_<SpectralGrid>(m, "SpectralGrid")
        .def(py::init<int, double>(), py::arg("N"), py::arg("L"))
        .def_readonly("N", &SpectralGrid::N)
        .def_readonly("L", &SpectralGrid::L)
        .def_readonly("Nz_half", &SpectralGrid::Nz_half)
        .def("kx", [](const SpectralGrid& g) {
            return py::array_t<double>(g.kx.size(), g.kx.data()); })
        .def("ky", [](const SpectralGrid& g) {
            return py::array_t<double>(g.ky.size(), g.ky.data()); })
        .def("kz", [](const SpectralGrid& g) {
            return py::array_t<double>(g.kz.size(), g.kz.data()); })
        .def("k2", [](const SpectralGrid& g) {
            py::array_t<double> arr({g.N, g.N, g.Nz_half});
            std::memcpy(arr.mutable_data(), g.k2.data(), g.k2.size()*sizeof(double));
            return arr; })
        .def("dealias_mask", [](const SpectralGrid& g) {
            py::array_t<double> arr({g.N, g.N, g.Nz_half});
            auto buf = arr.mutable_unchecked<3>();
            for (int i=0;i<g.N;++i)
              for (int j=0;j<g.N;++j)
                for (int k=0;k<g.Nz_half;++k)
                    buf(i,j,k)=(double)g.dealias_mask[g.index(i,j,k)];
            return arr; });

    // ---- Milestone 1: FFT3D ----
    py::class_<FFT3D>(m, "FFT3D")
        .def(py::init<int>(), py::arg("N"))
        .def("forward", [](FFT3D& f, py::array_t<double, py::array::c_style|py::array::forcecast> inp) {
            const int N=f.N(), Nzh=f.Nz_half();
            std::vector<cplx> out((size_t)N*N*Nzh);
            f.forward(inp.data(), out.data());
            py::array_t<cplx> res({N,N,Nzh});
            std::memcpy(res.mutable_data(), out.data(), out.size()*sizeof(cplx));
            return res; })
        .def("inverse", [](FFT3D& f, carray inp) {
            const int N=f.N();
            std::vector<double> out((size_t)N*N*N);
            f.inverse(reinterpret_cast<const cplx*>(inp.data()), out.data());
            py::array_t<double> res({N,N,N});
            std::memcpy(res.mutable_data(), out.data(), out.size()*sizeof(double));
            return res; });

    // ---- Milestone 2: projection and curl ----
    m.def("project_field", [](const SpectralGrid& g, carray ax, carray ay, carray az) {
        const size_t tot=(size_t)g.N*g.N*g.Nz_half;
        std::vector<cplx> Ax(ax.data(),ax.data()+tot);
        std::vector<cplx> Ay(ay.data(),ay.data()+tot);
        std::vector<cplx> Az(az.data(),az.data()+tot);
        project_field(g,Ax,Ay,Az);
        py::array_t<cplx> rx({g.N,g.N,g.Nz_half});
        py::array_t<cplx> ry({g.N,g.N,g.Nz_half});
        py::array_t<cplx> rz({g.N,g.N,g.Nz_half});
        std::memcpy(rx.mutable_data(),Ax.data(),tot*sizeof(cplx));
        std::memcpy(ry.mutable_data(),Ay.data(),tot*sizeof(cplx));
        std::memcpy(rz.mutable_data(),Az.data(),tot*sizeof(cplx));
        return py::make_tuple(rx,ry,rz);
    }, "Apply Coulomb gauge projection tensor P(k) to a Fourier-space vector field");

    m.def("spectral_curl", [](const SpectralGrid& g, carray ax, carray ay, carray az) {
        const size_t tot=(size_t)g.N*g.N*g.Nz_half;
        std::vector<cplx> Ax(ax.data(),ax.data()+tot);
        std::vector<cplx> Ay(ay.data(),ay.data()+tot);
        std::vector<cplx> Az(az.data(),az.data()+tot);
        std::vector<cplx> Bx,By,Bz;
        spectral_curl(g,Ax,Ay,Az,Bx,By,Bz);
        py::array_t<cplx> rx({g.N,g.N,g.Nz_half});
        py::array_t<cplx> ry({g.N,g.N,g.Nz_half});
        py::array_t<cplx> rz({g.N,g.N,g.Nz_half});
        std::memcpy(rx.mutable_data(),Bx.data(),tot*sizeof(cplx));
        std::memcpy(ry.mutable_data(),By.data(),tot*sizeof(cplx));
        std::memcpy(rz.mutable_data(),Bz.data(),tot*sizeof(cplx));
        return py::make_tuple(rx,ry,rz);
    }, "Compute B_hat = i k x A_hat (spectral curl)");

    // ---- Milestone 3: Hopf link ----
    m.def("compute_hopf_link",
        [](int N, double L, double R, double d, double a_core,
           double I0, double mu0, int n_quad) {
            std::vector<double> Ax,Ay,Az;
            hopf_link::compute(N,L,R,d,a_core,I0,mu0,n_quad,Ax,Ay,Az);
            py::array_t<double> rx({N,N,N});
            py::array_t<double> ry({N,N,N});
            py::array_t<double> rz({N,N,N});
            std::memcpy(rx.mutable_data(),Ax.data(),Ax.size()*sizeof(double));
            std::memcpy(ry.mutable_data(),Ay.data(),Ay.size()*sizeof(double));
            std::memcpy(rz.mutable_data(),Az.data(),Az.size()*sizeof(double));
            return py::make_tuple(rx,ry,rz);
        },
        "Regularized Biot-Savart Hopf link initial condition (OpenMP-parallel)",
        py::arg("N"), py::arg("L"), py::arg("R"), py::arg("d"), py::arg("a_core"),
        py::arg("I0")=1.0, py::arg("mu0")=1.0, py::arg("n_quad")=64);

    // ---- Milestone 5: ETDRK4 coefficient precomputation ----
    // Returns an ETDRK4Coeffs object with properties E, E2, f1, f2, f3.
    // L(k) = -alpha*k^2 - beta*k^4  (alpha=eta or nu, beta=eta4 or nu4).
    // All coefficients are computed via the M=32 Kassam-Trefethen contour
    // for every spectral mode without exception — no threshold switching.
    py::class_<ETDRK4Coeffs>(m, "ETDRK4Coeffs")
        .def_property_readonly("E", [](const ETDRK4Coeffs& c) {
            py::array_t<cplx> a({c.N,c.N,c.Nz_half});
            std::memcpy(a.mutable_data(),c.E.data(),c.E.size()*sizeof(cplx));
            return a; })
        .def_property_readonly("E2",[](const ETDRK4Coeffs& c) {
            py::array_t<cplx> a({c.N,c.N,c.Nz_half});
            std::memcpy(a.mutable_data(),c.E2.data(),c.E2.size()*sizeof(cplx));
            return a; })
        .def_property_readonly("f1",[](const ETDRK4Coeffs& c) {
            py::array_t<cplx> a({c.N,c.N,c.Nz_half});
            std::memcpy(a.mutable_data(),c.f1.data(),c.f1.size()*sizeof(cplx));
            return a; })
        .def_property_readonly("f2",[](const ETDRK4Coeffs& c) {
            py::array_t<cplx> a({c.N,c.N,c.Nz_half});
            std::memcpy(a.mutable_data(),c.f2.data(),c.f2.size()*sizeof(cplx));
            return a; })
        .def_property_readonly("f3",[](const ETDRK4Coeffs& c) {
            py::array_t<cplx> a({c.N,c.N,c.Nz_half});
            std::memcpy(a.mutable_data(),c.f3.data(),c.f3.size()*sizeof(cplx));
            return a; });

    m.def("precompute_etdrk4",
        [](const SpectralGrid& grid, double dt,
           double alpha, double beta, int M, double R) {
            return precompute_etdrk4(grid, dt, alpha, beta, M, R);
        },
        "Precompute ETDRK4 Kassam-Trefethen contour coefficients "
        "for L(k) = -alpha*k^2 - beta*k^4. "
        "No threshold switching: contour used for every mode including k=0.",
        py::arg("grid"), py::arg("dt"),
        py::arg("alpha"), py::arg("beta"),
        py::arg("M")=32, py::arg("R")=1.0);
}

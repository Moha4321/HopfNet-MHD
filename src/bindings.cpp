#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <complex>
#include <cstring>
#include <vector>

#include "spectral_grid.hpp"
#include "fft3d.hpp"
#include "projection.hpp"
#include "hopf_link.hpp"
#include "etdrk4_coeffs.hpp"
#include "null_finder.hpp" 
#include "point_cloud.hpp"  // <--- ADD THIS HERE

namespace py = pybind11;
using cplx   = std::complex<double>;
using carray  = py::array_t<cplx, py::array::c_style | py::array::forcecast>;

// ---- Legacy bridge test ----
Eigen::MatrixXd double_matrix(Eigen::Ref<const Eigen::MatrixXd> input) {
    return input * 2.0;
}

PYBIND11_MODULE(hopfnet_cpp, m) {
    m.doc() = "C++ Physics Engine Backend for HopfNet";
    m.def("double_matrix", &double_matrix, "A function that doubles an Eigen matrix");

    // ================================================================
    // Milestone 1: SpectralGrid
    // ================================================================
    py::class_<SpectralGrid>(m, "SpectralGrid")
        .def(py::init<int, double>(), py::arg("N"), py::arg("L"))
        .def_readonly("N",       &SpectralGrid::N)
        .def_readonly("L",       &SpectralGrid::L)
        .def_readonly("Nz_half", &SpectralGrid::Nz_half)
        .def("kx", [](const SpectralGrid& g) {
            return py::array_t<double>(g.kx.size(), g.kx.data()); })
        .def("ky", [](const SpectralGrid& g) {
            return py::array_t<double>(g.ky.size(), g.ky.data()); })
        .def("kz", [](const SpectralGrid& g) {
            return py::array_t<double>(g.kz.size(), g.kz.data()); })
        .def("k2", [](const SpectralGrid& g) {
            py::array_t<double> a({g.N, g.N, g.Nz_half});
            std::memcpy(a.mutable_data(), g.k2.data(), g.k2.size()*sizeof(double));
            return a; })
        .def("dealias_mask", [](const SpectralGrid& g) {
            py::array_t<double> a({g.N, g.N, g.Nz_half});
            auto b = a.mutable_unchecked<3>();
            for (int i=0;i<g.N;++i)
                for (int j=0;j<g.N;++j)
                    for (int k=0;k<g.Nz_half;++k)
                        b(i,j,k)=(double)g.dealias_mask[g.index(i,j,k)];
            return a; });

    // ================================================================
    // Milestone 1: FFT3D
    // ================================================================
    py::class_<FFT3D>(m, "FFT3D")
        .def(py::init<int>(), py::arg("N"))
        .def("forward", [](FFT3D& f,
            py::array_t<double, py::array::c_style|py::array::forcecast> inp) {
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

    // ================================================================
    // Milestone 2: Coulomb gauge projection and spectral curl
    // ================================================================
    m.def("project_field",
        [](const SpectralGrid& g, carray ax, carray ay, carray az) {
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
        }, "Apply Coulomb gauge projection tensor P(k)");

    m.def("spectral_curl",
        [](const SpectralGrid& g, carray ax, carray ay, carray az) {
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

    // ================================================================
    // Milestone 3: Regularized Biot-Savart Hopf link IC
    // ================================================================
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
        "Regularized Biot-Savart Hopf link initial condition",
        py::arg("N"), py::arg("L"), py::arg("R"), py::arg("d"), py::arg("a_core"),
        py::arg("I0")=1.0, py::arg("mu0")=1.0, py::arg("n_quad")=64);

    // ================================================================
    // Milestone 5: ETDRK4 Kassam-Trefethen contour coefficients
    //
    // Two independent ETDCoeffs objects will be instantiated at t=0:
    //   coeffs_A: alpha=eta,  beta=eta4  (induction equation)
    //   coeffs_v: alpha=nu,   beta=nu4   (momentum equation)
    //
    // ALL modes including k=0 use the contour integral — no switching.
    // ================================================================
    py::class_<ETDCoeffs>(m, "ETDCoeffs")
        .def(py::init<const SpectralGrid&, double, double, double, int>(),
             py::arg("grid"), py::arg("dt"),
             py::arg("alpha"), py::arg("beta"), py::arg("M")=32,
             "Precompute ETDRK4 coefficients via Kassam-Trefethen contour "
             "for L(k) = -alpha*k^2 - beta*k^4. No threshold switching.")
        .def("e1", [](const ETDCoeffs& c, const SpectralGrid& g) {
            py::array_t<double> a({g.N,g.N,g.Nz_half});
            std::memcpy(a.mutable_data(),c.e1.data(),c.e1.size()*sizeof(double));
            return a; }, py::arg("grid"))
        .def("e2", [](const ETDCoeffs& c, const SpectralGrid& g) {
            py::array_t<double> a({g.N,g.N,g.Nz_half});
            std::memcpy(a.mutable_data(),c.e2.data(),c.e2.size()*sizeof(double));
            return a; }, py::arg("grid"))
        .def("f1", [](const ETDCoeffs& c, const SpectralGrid& g) {
            py::array_t<double> a({g.N,g.N,g.Nz_half});
            std::memcpy(a.mutable_data(),c.f1.data(),c.f1.size()*sizeof(double));
            return a; }, py::arg("grid"))
        .def("f2", [](const ETDCoeffs& c, const SpectralGrid& g) {
            py::array_t<double> a({g.N,g.N,g.Nz_half});
            std::memcpy(a.mutable_data(),c.f2.data(),c.f2.size()*sizeof(double));
            return a; }, py::arg("grid"))
        .def("f3", [](const ETDCoeffs& c, const SpectralGrid& g) {
            py::array_t<double> a({g.N,g.N,g.Nz_half});
            std::memcpy(a.mutable_data(),c.f3.data(),c.f3.size()*sizeof(double));
            return a; }, py::arg("grid"));

    // ================================================================
    // Milestone 9: Haynes-Parnell Magnetic Null Finder
    // ================================================================
    m.def("find_nulls",
        [](int N, double L, 
           py::array_t<double, py::array::c_style | py::array::forcecast> bx,
           py::array_t<double, py::array::c_style | py::array::forcecast> by,
           py::array_t<double, py::array::c_style | py::array::forcecast> bz) {
            
            std::vector<MagneticNull> nulls = null_finder::find_nulls(
                N, L, bx.data(), by.data(), bz.data()
            );

            // Convert to numpy arrays
            int M = nulls.size();
            py::array_t<double> pos({M, 3});
            py::array_t<int> types(M);
            
            auto pos_r = pos.mutable_unchecked<2>();
            auto types_r = types.mutable_unchecked<1>();
            
            for (int i = 0; i < M; ++i) {
                pos_r(i, 0) = nulls[i].x;
                pos_r(i, 1) = nulls[i].y;
                pos_r(i, 2) = nulls[i].z;
                types_r(i) = nulls[i].type;
            }
            
            return py::make_tuple(pos, types);
        }, 
        "Find and classify magnetic nulls using Newton-Raphson",
        py::arg("N"), py::arg("L"), py::arg("bx"), py::arg("by"), py::arg("bz"));

// ================================================================
    // Milestone 11: Current-Sheet Point Cloud Extractor
    // ================================================================
    m.def("extract_point_cloud",
        [](const SpectralGrid& grid, FFT3D& fft,
           carray jx_hat, carray jy_hat, carray jz_hat, 
           double threshold) {
            
            std::vector<double> raw_data = point_cloud::extract(
                grid, fft, jx_hat.data(), jy_hat.data(), jz_hat.data(), threshold
            );

            int M = raw_data.size() / 9;
            py::array_t<double> out({M, 9});
            std::memcpy(out.mutable_data(), raw_data.data(), raw_data.size() * sizeof(double));
            
            return out;
        }, 
        "Extract active current sheet points and their symmetric Shape Operator",
        py::arg("grid"), py::arg("fft"), 
        py::arg("jx_hat"), py::arg("jy_hat"), py::arg("jz_hat"), 
        py::arg("threshold") = 0.6);
}
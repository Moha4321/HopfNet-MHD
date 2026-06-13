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

namespace py = pybind11;

// A simple function to test Eigen and PyBind11 memory mapping
Eigen::MatrixXd double_matrix(Eigen::Ref<const Eigen::MatrixXd> input) {
    return input * 2.0;
}

PYBIND11_MODULE(hopfnet_cpp, m) {
    m.doc() = "C++ Physics Engine Backend for HopfNet";
    m.def("double_matrix", &double_matrix, "A function that doubles an Eigen matrix");

    // ---- SpectralGrid: wavevectors, |k|^2, dealias mask ----
    py::class_<SpectralGrid>(m, "SpectralGrid")
        .def(py::init<int, double>(), py::arg("N"), py::arg("L"))
        .def_readonly("N", &SpectralGrid::N)
        .def_readonly("L", &SpectralGrid::L)
        .def_readonly("Nz_half", &SpectralGrid::Nz_half)
        .def("kx", [](const SpectralGrid& g) {
            return py::array_t<double>(g.kx.size(), g.kx.data());
        })
        .def("ky", [](const SpectralGrid& g) {
            return py::array_t<double>(g.ky.size(), g.ky.data());
        })
        .def("kz", [](const SpectralGrid& g) {
            return py::array_t<double>(g.kz.size(), g.kz.data());
        })
        .def("k2", [](const SpectralGrid& g) {
            py::array_t<double> arr({g.N, g.N, g.Nz_half});
            std::memcpy(arr.mutable_data(), g.k2.data(), g.k2.size() * sizeof(double));
            return arr;
        })
        .def("dealias_mask", [](const SpectralGrid& g) {
            py::array_t<double> arr({g.N, g.N, g.Nz_half});
            auto buf = arr.mutable_unchecked<3>();
            for (int i = 0; i < g.N; ++i)
                for (int j = 0; j < g.N; ++j)
                    for (int k = 0; k < g.Nz_half; ++k)
                        buf(i, j, k) = (double)g.dealias_mask[g.index(i, j, k)];
            return arr;
        });

    // ---- FFT3D: real-to-complex 3D FFT pair ----
    py::class_<FFT3D>(m, "FFT3D")
        .def(py::init<int>(), py::arg("N"))
        .def("forward", [](FFT3D& f, py::array_t<double, py::array::c_style | py::array::forcecast> input) {
            const int N = f.N(), Nzh = f.Nz_half();
            std::vector<std::complex<double>> out((size_t)N * N * Nzh);
            f.forward(input.data(), out.data());

            py::array_t<std::complex<double>> result({N, N, Nzh});
            std::memcpy(result.mutable_data(), out.data(), out.size() * sizeof(std::complex<double>));
            return result;
        })
        .def("inverse", [](FFT3D& f, py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> input) {
            const int N = f.N();
            std::vector<double> out((size_t)N * N * N);
            f.inverse(reinterpret_cast<const std::complex<double>*>(input.data()), out.data());

            py::array_t<double> result({N, N, N});
            std::memcpy(result.mutable_data(), out.data(), out.size() * sizeof(double));
            return result;
        });

    // ---- Milestone 2: Coulomb gauge projection and spectral curl ----
    using carray = py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>;

    m.def("project_field", [](const SpectralGrid& grid, carray ax, carray ay, carray az) {
        const size_t total = (size_t)grid.N * grid.N * grid.Nz_half;
        std::vector<cplx> Ax(ax.data(), ax.data() + total);
        std::vector<cplx> Ay(ay.data(), ay.data() + total);
        std::vector<cplx> Az(az.data(), az.data() + total);

        project_field(grid, Ax, Ay, Az);

        py::array_t<std::complex<double>> rx({grid.N, grid.N, grid.Nz_half});
        py::array_t<std::complex<double>> ry({grid.N, grid.N, grid.Nz_half});
        py::array_t<std::complex<double>> rz({grid.N, grid.N, grid.Nz_half});
        std::memcpy(rx.mutable_data(), Ax.data(), total * sizeof(cplx));
        std::memcpy(ry.mutable_data(), Ay.data(), total * sizeof(cplx));
        std::memcpy(rz.mutable_data(), Az.data(), total * sizeof(cplx));
        return py::make_tuple(rx, ry, rz);
    }, "Apply the Coulomb gauge projection tensor P(k) to a vector field in Fourier space");

    m.def("spectral_curl", [](const SpectralGrid& grid, carray ax, carray ay, carray az) {
        const size_t total = (size_t)grid.N * grid.N * grid.Nz_half;
        std::vector<cplx> Ax(ax.data(), ax.data() + total);
        std::vector<cplx> Ay(ay.data(), ay.data() + total);
        std::vector<cplx> Az(az.data(), az.data() + total);
        std::vector<cplx> Bx, By, Bz;

        spectral_curl(grid, Ax, Ay, Az, Bx, By, Bz);

        py::array_t<std::complex<double>> rx({grid.N, grid.N, grid.Nz_half});
        py::array_t<std::complex<double>> ry({grid.N, grid.N, grid.Nz_half});
        py::array_t<std::complex<double>> rz({grid.N, grid.N, grid.Nz_half});
        std::memcpy(rx.mutable_data(), Bx.data(), total * sizeof(cplx));
        std::memcpy(ry.mutable_data(), By.data(), total * sizeof(cplx));
        std::memcpy(rz.mutable_data(), Bz.data(), total * sizeof(cplx));
        return py::make_tuple(rx, ry, rz);
    }, "Compute B_hat = i k x A_hat (spectral curl)");

    // ---- Milestone 3: Hopf link initial condition (regularized Biot-Savart) ----
    m.def("compute_hopf_link", [](int N, double L, double R, double d, double a_core,
                                   double I0, double mu0, int n_quad) {
        std::vector<double> Ax, Ay, Az;
        hopf_link::compute(N, L, R, d, a_core, I0, mu0, n_quad, Ax, Ay, Az);

        py::array_t<double> rx({N, N, N});
        py::array_t<double> ry({N, N, N});
        py::array_t<double> rz({N, N, N});
        std::memcpy(rx.mutable_data(), Ax.data(), Ax.size() * sizeof(double));
        std::memcpy(ry.mutable_data(), Ay.data(), Ay.size() * sizeof(double));
        std::memcpy(rz.mutable_data(), Az.data(), Az.size() * sizeof(double));
        return py::make_tuple(rx, ry, rz);
    }, "Regularized Biot-Savart vector potential for the magnetic Hopf link initial condition",
       py::arg("N"), py::arg("L"), py::arg("R"), py::arg("d"), py::arg("a_core"),
       py::arg("I0") = 1.0, py::arg("mu0") = 1.0, py::arg("n_quad") = 64);
}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

namespace py = pybind11;

// A simple function to test Eigen and PyBind11 memory mapping
Eigen::MatrixXd double_matrix(Eigen::Ref<const Eigen::MatrixXd> input) {
    return input * 2.0;
}

// Create the Python module named 'hopfnet_cpp'
PYBIND11_MODULE(hopfnet_cpp, m) {
    m.doc() = "C++ Physics Engine Backend for HopfNet"; // Optional module docstring
    m.def("double_matrix", &double_matrix, "A function that doubles an Eigen matrix");
}

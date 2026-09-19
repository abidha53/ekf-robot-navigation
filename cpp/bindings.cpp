#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "ekf.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ekf_cpp, m) {
    m.doc() = "C++ Extended Kalman Filter for 2D differential-drive robot localization";

    py::class_<EKF>(m, "EKF")
        .def(py::init<const Eigen::Vector3d&, const Eigen::Matrix3d&>(),
             py::arg("initial_state"), py::arg("initial_covariance"),
             "Construct an EKF from an initial [x, y, theta] state and 3x3 covariance.")
        .def("predict", &EKF::predict,
             py::arg("v"), py::arg("omega"), py::arg("dt"),
             "Propagate the belief forward using commanded (noisy-odometry) velocities.")
        .def("update", &EKF::update,
             py::arg("range"), py::arg("bearing"), py::arg("landmark_x"), py::arg("landmark_y"),
             "Fuse one landmark range-bearing measurement into the belief.")
        .def("get_state", &EKF::get_state,
             "Return the current [x, y, theta] state estimate as a numpy array.")
        .def("get_covariance", &EKF::get_covariance,
             "Return the current 3x3 covariance matrix as a numpy array.");
}

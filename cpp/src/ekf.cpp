#include "ekf.hpp"

#include <cmath>

namespace {

constexpr double kPi = 3.14159265358979323846;
}

EKF::EKF(const Eigen::Vector3d& initial_state,
         const Eigen::Matrix3d& initial_covariance)

    : state_(initial_state), covariance_(initial_covariance) {}

void EKF::predict(double v, double omega, double dt) {
    const double theta = state_(2);
    const double cos_theta = std::cos(theta);
    const double sin_theta = std::sin(theta);

    state_(0) += v * cos_theta * dt;
    state_(1) += v * sin_theta * dt;
    state_(2) = normalize_angle(state_(2) + omega * dt);

    Eigen::Matrix3d G = Eigen::Matrix3d::Identity();
    G(0, 2) = -v * sin_theta * dt;
    G(1, 2) = v * cos_theta * dt;

    Eigen::Matrix<double, 3, 2> V;
    V << cos_theta * dt, 0.0,
         sin_theta * dt, 0.0,
         0.0,            dt;

    Eigen::Matrix2d M;
    M << kAlpha1 * v * v + kAlpha2 * omega * omega, 0.0,
         0.0,                                       kAlpha3 * v * v + kAlpha4 * omega * omega;

    covariance_ = G * covariance_ * G.transpose() + V * M * V.transpose();
}

void EKF::update(double range, double bearing, double landmark_x, double landmark_y) {

    const double dx = landmark_x - state_(0);
    const double dy = landmark_y - state_(1);
    const double q = dx * dx + dy * dy;
    const double range_expected = std::sqrt(q);
    const double bearing_expected =
        normalize_angle(std::atan2(dy, dx) - state_(2));

    Eigen::Vector2d innovation;
    innovation(0) = range - range_expected;
    innovation(1) = normalize_angle(bearing - bearing_expected);

    Eigen::Matrix<double, 2, 3> H;
    H << -dx / range_expected, -dy / range_expected, 0.0,
          dy / q,              -dx / q,              -1.0;

    Eigen::Matrix2d Q;
    Q << kSigmaRange * kSigmaRange, 0.0,
         0.0,                       kSigmaBearing * kSigmaBearing;

    const Eigen::Matrix2d S = H * covariance_ * H.transpose() + Q;
    const Eigen::Matrix<double, 3, 2> K = covariance_ * H.transpose() * S.inverse();

    state_ = state_ + K * innovation;
    state_(2) = normalize_angle(state_(2));

    covariance_ = (Eigen::Matrix3d::Identity() - K * H) * covariance_;
}

Eigen::Vector3d EKF::get_state() const {
    return state_;
}

Eigen::Matrix3d EKF::get_covariance() const {
    return covariance_;
}

double EKF::normalize_angle(double angle) {

    while (angle > kPi) angle -= 2.0 * kPi;
    while (angle <= -kPi) angle += 2.0 * kPi;
    return angle;
}

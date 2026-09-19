#pragma once

#include <Eigen/Dense>

class EKF {
public:

    EKF(const Eigen::Vector3d& initial_state,
        const Eigen::Matrix3d& initial_covariance);

    void predict(double v, double omega, double dt);

    void update(double range, double bearing, double landmark_x, double landmark_y);

    Eigen::Vector3d get_state() const;
    Eigen::Matrix3d get_covariance() const;

private:

    Eigen::Vector3d state_;
    Eigen::Matrix3d covariance_;

    static constexpr double kAlpha1 = 0.02;
    static constexpr double kAlpha2 = 0.01;
    static constexpr double kAlpha3 = 0.01;
    static constexpr double kAlpha4 = 0.05;

    static constexpr double kSigmaRange = 0.7;
    static constexpr double kSigmaBearing = 0.05;

    static double normalize_angle(double angle);
};

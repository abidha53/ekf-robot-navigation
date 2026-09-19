#include <gtest/gtest.h>

#include "ekf.hpp"

TEST(EKFPredict, MovesStateAlongHeadingAndGrowsCovariance) {
    Eigen::Vector3d initial_state(0.0, 0.0, 0.0);
    Eigen::Matrix3d initial_covariance = Eigen::Matrix3d::Identity() * 0.01;

    EKF ekf(initial_state, initial_covariance);
    const double covariance_trace_before = ekf.get_covariance().trace();

    ekf.predict(1.0, 0.0, 1.0);

    Eigen::Vector3d state = ekf.get_state();
    EXPECT_NEAR(state(0), 1.0, 1e-9);
    EXPECT_NEAR(state(1), 0.0, 1e-9);
    EXPECT_NEAR(state(2), 0.0, 1e-9);

    const double covariance_trace_after = ekf.get_covariance().trace();
    EXPECT_GT(covariance_trace_after, covariance_trace_before);
}

TEST(EKFUpdate, MovesEstimateTowardTruthAndShrinksCovariance) {
    const Eigen::Vector3d true_pose(0.0, 0.0, 0.0);
    const double landmark_x = 5.0;
    const double landmark_y = 0.0;

    Eigen::Vector3d biased_estimate(0.5, 0.5, 0.0);
    Eigen::Matrix3d initial_covariance = Eigen::Matrix3d::Identity() * 1.0;

    EKF ekf(biased_estimate, initial_covariance);
    const double error_before = (biased_estimate - true_pose).norm();
    const double covariance_trace_before = ekf.get_covariance().trace();

    ekf.update(5.0, 0.0, landmark_x, landmark_y);

    Eigen::Vector3d state_after = ekf.get_state();
    const double error_after = (state_after - true_pose).norm();
    EXPECT_LT(error_after, error_before);

    const double covariance_trace_after = ekf.get_covariance().trace();
    EXPECT_LT(covariance_trace_after, covariance_trace_before);
}

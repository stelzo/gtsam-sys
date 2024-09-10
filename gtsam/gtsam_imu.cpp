#include "gtsam_imu.hpp"
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/PreintegrationBase.h>
#include <gtsam/navigation/ImuFactor.h>

#include <optional>

#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/PreintegrationCombinedParams.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/inference/Symbol.h>

#include <Eigen/Dense>

typedef struct
{
    gtsam::imuBias::ConstantBias biases_;
    gtsam::PreintegratedImuMeasurements *integrator_;
    gtsam::Vector3 angular_velocity_;
    gtsam::NavState start_pose_;
} State;

void *init_integrate_state(double gravity, double accel_noise, double gyro_noise, double accel_bias_noise, double gyro_bias_noise, double integration_noise, Vector3 bias_acc, Vector3 bias_gyro)
{
    gtsam::Matrix3 I_3x3 = gtsam::Matrix3::Identity();

    auto imuParams = gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedU(gravity);
    imuParams->accelerometerCovariance = I_3x3 * pow(accel_noise, 2); // accelerometer measurement noise
    imuParams->gyroscopeCovariance = I_3x3 * pow(gyro_noise, 2);      // gyroscope measurement noise
    imuParams->integrationCovariance = I_3x3 * integration_noise;     // integration uncertainty
    imuParams->biasAccCovariance = I_3x3 * pow(accel_bias_noise, 2);  // accelerometer bias noise
    imuParams->biasOmegaCovariance = I_3x3 * pow(gyro_bias_noise, 2); // gyroscope bias noise

    auto imuBias = gtsam::imuBias::ConstantBias(gtsam::Vector3(bias_acc.x, bias_acc.y, bias_acc.z), gtsam::Vector3(bias_gyro.x, bias_gyro.y, bias_gyro.z));
    gtsam::PreintegratedCombinedMeasurements preintegratedMeasurements(imuParams, imuBias);
    auto integrator = new gtsam::PreintegratedImuMeasurements(imuParams);

    State *state = new State();
    state->integrator_ = integrator;
    return state;
}

void destroy_integrate_state(void *state)
{
    State *state_ = static_cast<State *>(state);
    delete state_->integrator_;
    delete state_;
}

void add_measurement(void *state, double dt, const Vector3 &linear_acceleration, const Vector3 &angular_velocity)
{
    State *state_ = static_cast<State *>(state);
    gtsam::Vector3 acc(linear_acceleration.x, linear_acceleration.y, linear_acceleration.z);
    gtsam::Vector3 gyro(angular_velocity.x, angular_velocity.y, angular_velocity.z);
    state_->integrator_->integrateMeasurement(acc, gyro, dt);

    // propagate angular velocity in body frame
    state_->angular_velocity_ = state_->biases_.correctGyroscope(gyro);
}

void getState(const void *state,
              Transformation *transform,
              Vector3 *linear_velocity,
              Vector3 *angular_velocity)
{
    const State *state_ = static_cast<const State *>(state);

    // 4. Predict the state (pose, velocity) at the end of integration
    gtsam::Pose3 initialPose(gtsam::Rot3::Identity(), gtsam::Point3(0, 0, 0));
    gtsam::Vector3 initialVelocity(0.0, 0.0, 0.0);
    gtsam::NavState initialState(initialPose, initialVelocity);

    // Apply the preintegrated measurements to propagate the state
    gtsam::NavState predictedState = state_->integrator_->predict(initialState, state_->biases_);

    // 5. Extract the results
    gtsam::Pose3 predictedPose = predictedState.pose();
    gtsam::Vector3 predictedVelocity = predictedState.velocity();

    // Set output variables with the predicted state
    gtsam::Vector3 position_at_target = predictedPose.translation();

    auto quat = predictedPose.rotation().toQuaternion();

    transform->position = Vector3{position_at_target.x(), position_at_target.y(), position_at_target.z()};
    transform->rotation = Quaternion{quat.x(), quat.y(), quat.z(), quat.w()};
    *linear_velocity = Vector3{predictedVelocity.x(), predictedVelocity.y(), predictedVelocity.z()};
    *angular_velocity = Vector3{state_->angular_velocity_.x(), state_->angular_velocity_.y(), state_->angular_velocity_.z()}; // Directly using gyroscope reading for angular velocity
}

void resetState(void *state, const Transformation &transform, const Vector3 &linear_velocity, const Vector3 &angular_velocity)
{
    State *state_ = static_cast<State *>(state);

    state_->integrator_->resetIntegration();

    Eigen::Quaterniond q(transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z);
    gtsam::Rot3 rot(q.toRotationMatrix());

    gtsam::Vector3 pos = gtsam::Vector3(transform.position.x, transform.position.y, transform.position.z);
    gtsam::Vector3 linear_velocity_ = gtsam::Vector3(linear_velocity.x, linear_velocity.y, linear_velocity.z);

    state_->start_pose_ = gtsam::NavState::Create(rot,
                                                  pos,
                                                  linear_velocity_,
                                                  std::nullopt,
                                                  std::nullopt,
                                                  std::nullopt);

    state_->angular_velocity_.x() = angular_velocity.x;
    state_->angular_velocity_.y() = angular_velocity.y;
    state_->angular_velocity_.z() = angular_velocity.z;
}

void resetBias(void *state, const Vector3 &imu_angular_velocity_bias,
               const Vector3 &imu_linear_acceleration_bias)
{
    State *state_ = static_cast<State *>(state);
    gtsam::Vector3 imu_linear_acceleration_bias_ = gtsam::Vector3(imu_linear_acceleration_bias.x,
                                                                  imu_linear_acceleration_bias.y,
                                                                  imu_linear_acceleration_bias.z);
    gtsam::Vector3 imu_angular_velocity_bias_ = gtsam::Vector3(imu_angular_velocity_bias.x,
                                                               imu_angular_velocity_bias.y,
                                                               imu_angular_velocity_bias.z);

    state_->biases_ = gtsam::imuBias::ConstantBias(imu_linear_acceleration_bias_,
                                                   imu_angular_velocity_bias_);
}

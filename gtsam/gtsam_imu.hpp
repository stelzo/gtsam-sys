#ifndef GTSAM_IMU_H
#define GTSAM_IMU_H

typedef struct
{
    double x;
    double y;
    double z;
    double w;
} Quaternion;

typedef struct
{
    double x;
    double y;
    double z;
} Vector3;

typedef struct
{
    Vector3 position;
    Quaternion rotation;
} Transformation;

void *init_integrate_state(double gravity, double accel_noise, double gyro_noise, double accel_bias_noise, double gyro_bias_noise, double integration_noise, Vector3 bias_acc, Vector3 bias_gyro);
void destroy_integrate_state(void *state);

void add_measurement(void *state, double dt, const Vector3 &linear_acceleration, const Vector3 &angular_velocity);

void getState(const void *state,
              Transformation *transform,
              Vector3 *linear_velocity,
              Vector3 *angular_velocity);

void resetState(void *state, const Transformation &transform, const Vector3 &linear_velocity, const Vector3 &angular_velocity);

void resetBias(void *state, const Vector3 &imu_angular_velocity_bias,
               const Vector3 &imu_linear_acceleration_bias);

#endif // GTSAM_IMU_H
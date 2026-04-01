#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![doc = "Bridge-only migration crate. Prefer the pure-Rust API in `gtsam-rs`."]

pub mod config;

use nalgebra as na;

#[deprecated(
    since = "0.1.0",
    note = "gtsam-sys is bridge-only during migration; prefer gtsam-rs::navigation::IntegrationState"
)]
pub struct IntegrationState {
    pub translation: na::Vector3<f64>,
    pub rotation: na::UnitQuaternion<f64>,
    pub linear_velocity: na::Vector3<f64>,
    pub angular_velocity: na::Vector3<f64>,
}

#[cfg(feature = "pure-rust-navigation")]
#[allow(deprecated)]
mod backend {
    use super::{config::ImuParameters, IntegrationState};
    use gtsam_rs_navigation as nav;
    use nalgebra as na;

    #[derive(Debug)]
    pub struct ImuIntegration {
        inner: nav::ImuIntegration,
    }

    impl ImuIntegration {
        pub fn new(params: &ImuParameters) -> Self {
            let params = nav::ImuParameters {
                gravity: params.gravity,
                accel_noise: params.accel_noise,
                gyro_noise: params.gyro_noise,
                accel_bias_noise: params.accel_bias_noise,
                gyro_bias_noise: params.gyro_bias_noise,
                integration_noise: params.integration_noise,
                bias_acc: params.bias_acc,
                bias_gyro: params.bias_gyro,
            };
            Self {
                inner: nav::ImuIntegration::new(&params),
            }
        }

        pub fn add_imu(
            &mut self,
            dt: f64,
            linear_acceleration: na::Vector3<f64>,
            angular_velocity: na::Vector3<f64>,
        ) {
            self.inner
                .add_imu(dt, linear_acceleration, angular_velocity)
                .expect("dt must be positive");
        }

        pub fn get_state(&self) -> IntegrationState {
            let state = self.inner.get_state();
            IntegrationState {
                translation: state.translation,
                rotation: state.rotation,
                linear_velocity: state.linear_velocity,
                angular_velocity: state.angular_velocity,
            }
        }

        pub fn reset_state(
            &mut self,
            transform: na::Isometry3<f64>,
            linear_velocity: na::Vector3<f64>,
            angular_velocity: na::Vector3<f64>,
        ) {
            self.inner
                .reset_state(transform, linear_velocity, angular_velocity);
        }

        pub fn reset_bias(
            &mut self,
            imu_angular_velocity_bias: na::Vector3<f64>,
            imu_linear_acceleration_bias: na::Vector3<f64>,
        ) {
            self.inner
                .reset_bias(imu_angular_velocity_bias, imu_linear_acceleration_bias);
        }
    }

    pub use ImuIntegration as PublicImuIntegration;
}

#[cfg(not(feature = "pure-rust-navigation"))]
#[allow(deprecated)]
mod backend {
    use std::borrow::{Borrow, BorrowMut};

    use super::{config::ImuParameters, IntegrationState};
    use nalgebra::{self as na, UnitQuaternion};

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

    #[derive(Debug)]
    pub struct ImuIntegration {
        state: *mut std::os::raw::c_void,
    }

    unsafe impl Send for ImuIntegration {}
    unsafe impl Sync for ImuIntegration {}

    impl Default for Vector3 {
        fn default() -> Self {
            Self {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            }
        }
    }

    impl Default for Quaternion {
        fn default() -> Self {
            Self {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            }
        }
    }

    impl Default for Transformation {
        fn default() -> Self {
            Self {
                position: Vector3::default(),
                rotation: Quaternion::default(),
            }
        }
    }

    impl ImuIntegration {
        pub fn new(params: &ImuParameters) -> Self {
            let bias_acc = Vector3 {
                x: params.bias_acc.x,
                y: params.bias_acc.y,
                z: params.bias_acc.z,
            };
            let bias_gyro = Vector3 {
                x: params.bias_gyro.x,
                y: params.bias_gyro.y,
                z: params.bias_gyro.z,
            };
            let state = unsafe {
                init_integrate_state(
                    params.gravity,
                    params.accel_noise,
                    params.gyro_noise,
                    params.accel_bias_noise,
                    params.gyro_bias_noise,
                    params.integration_noise,
                    bias_acc,
                    bias_gyro,
                )
            };
            Self { state }
        }

        pub fn add_imu(
            &mut self,
            dt: f64,
            linear_acceleration: na::Vector3<f64>,
            angular_velocity: na::Vector3<f64>,
        ) {
            assert!(dt > 0.0, "dt must be positive");
            let linear_acceleration = Vector3 {
                x: linear_acceleration.x,
                y: linear_acceleration.y,
                z: linear_acceleration.z,
            };

            let angular_velocity = Vector3 {
                x: angular_velocity.x,
                y: angular_velocity.y,
                z: angular_velocity.z,
            };

            unsafe {
                add_measurement(
                    self.state,
                    dt,
                    linear_acceleration.borrow(),
                    angular_velocity.borrow(),
                )
            }
        }

        pub fn get_state(&self) -> IntegrationState {
            let mut linear_velocity = Vector3::default();
            let mut angular_velocity = Vector3::default();
            let mut transform = Transformation::default();

            unsafe {
                getState(
                    self.state,
                    transform.borrow_mut(),
                    linear_velocity.borrow_mut(),
                    angular_velocity.borrow_mut(),
                )
            }

            IntegrationState {
                translation: na::Vector3::new(
                    transform.position.x,
                    transform.position.y,
                    transform.position.z,
                ),
                rotation: UnitQuaternion::new_normalize(na::Quaternion::new(
                    transform.rotation.w,
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                )),
                linear_velocity: na::Vector3::new(
                    linear_velocity.x,
                    linear_velocity.y,
                    linear_velocity.z,
                ),
                angular_velocity: na::Vector3::new(
                    angular_velocity.x,
                    angular_velocity.y,
                    angular_velocity.z,
                ),
            }
        }

        pub fn reset_state(
            &mut self,
            transform: na::Isometry3<f64>,
            linear_velocity: na::Vector3<f64>,
            angular_velocity: na::Vector3<f64>,
        ) {
            let transform = Transformation {
                position: Vector3 {
                    x: transform.translation.vector.x,
                    y: transform.translation.vector.y,
                    z: transform.translation.vector.z,
                },
                rotation: Quaternion {
                    x: transform.rotation.coords.x,
                    y: transform.rotation.coords.y,
                    z: transform.rotation.coords.z,
                    w: transform.rotation.coords.w,
                },
            };

            let linear_velocity = Vector3 {
                x: linear_velocity.x,
                y: linear_velocity.y,
                z: linear_velocity.z,
            };

            let angular_velocity = Vector3 {
                x: angular_velocity.x,
                y: angular_velocity.y,
                z: angular_velocity.z,
            };

            unsafe {
                resetState(
                    self.state,
                    transform.borrow(),
                    linear_velocity.borrow(),
                    angular_velocity.borrow(),
                )
            }
        }

        #[allow(dead_code)]
        pub fn reset_bias(
            &mut self,
            imu_angular_velocity_bias: na::Vector3<f64>,
            imu_linear_acceleration_bias: na::Vector3<f64>,
        ) {
            let imu_angular_velocity_bias = Vector3 {
                x: imu_angular_velocity_bias.x,
                y: imu_angular_velocity_bias.y,
                z: imu_angular_velocity_bias.z,
            };

            let imu_linear_acceleration_bias = Vector3 {
                x: imu_linear_acceleration_bias.x,
                y: imu_linear_acceleration_bias.y,
                z: imu_linear_acceleration_bias.z,
            };

            unsafe {
                resetBias(
                    self.state,
                    imu_angular_velocity_bias.borrow(),
                    imu_linear_acceleration_bias.borrow(),
                )
            }
        }
    }

    impl Drop for ImuIntegration {
        fn drop(&mut self) {
            unsafe {
                destroy_integrate_state(self.state);
            }
        }
    }

    pub use ImuIntegration as PublicImuIntegration;
}

#[deprecated(
    since = "0.1.0",
    note = "gtsam-sys is bridge-only during migration; prefer gtsam-rs::navigation::ImuIntegration"
)]
pub use backend::PublicImuIntegration as ImuIntegration;

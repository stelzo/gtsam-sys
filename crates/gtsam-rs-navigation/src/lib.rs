#![forbid(unsafe_code)]

use nalgebra::{Isometry3, UnitQuaternion, Vector3};

#[derive(Debug, Clone)]
pub struct ImuParameters {
    pub gravity: f64,
    pub accel_noise: f64,
    pub gyro_noise: f64,
    pub accel_bias_noise: f64,
    pub gyro_bias_noise: f64,
    pub integration_noise: f64,
    pub bias_acc: Vector3<f64>,
    pub bias_gyro: Vector3<f64>,
}

impl Default for ImuParameters {
    fn default() -> Self {
        Self {
            gravity: 9.81,
            accel_noise: 1e-3,
            gyro_noise: 1e-5,
            accel_bias_noise: 1e-5,
            gyro_bias_noise: 1e-5,
            integration_noise: 1e-8,
            bias_acc: Vector3::zeros(),
            bias_gyro: Vector3::zeros(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ImuBias {
    pub accel: Vector3<f64>,
    pub gyro: Vector3<f64>,
}

#[derive(Clone, Debug)]
pub struct NavState {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub rotation: UnitQuaternion<f64>,
}

#[derive(Clone, Debug)]
pub struct PreintegratedImuMeasurements {
    pub delta_t: f64,
}

#[derive(Clone, Debug)]
pub struct CombinedImuFactor;

#[derive(Clone, Debug)]
pub struct IntegrationState {
    pub translation: Vector3<f64>,
    pub rotation: UnitQuaternion<f64>,
    pub linear_velocity: Vector3<f64>,
    pub angular_velocity: Vector3<f64>,
}

impl Default for IntegrationState {
    fn default() -> Self {
        Self {
            translation: Vector3::zeros(),
            rotation: UnitQuaternion::identity(),
            linear_velocity: Vector3::zeros(),
            angular_velocity: Vector3::zeros(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavigationError {
    NonPositiveDt,
}

impl std::fmt::Display for NavigationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NavigationError::NonPositiveDt => f.write_str("dt must be positive"),
        }
    }
}

impl std::error::Error for NavigationError {}

#[derive(Debug, Clone)]
pub struct ImuIntegration {
    params: ImuParameters,
    state: IntegrationState,
}

impl ImuIntegration {
    pub fn new(params: &ImuParameters) -> Self {
        Self {
            params: params.clone(),
            state: IntegrationState::default(),
        }
    }

    pub fn add_imu(
        &mut self,
        dt: f64,
        linear_acceleration: Vector3<f64>,
        angular_velocity: Vector3<f64>,
    ) -> Result<(), NavigationError> {
        if dt <= 0.0 {
            return Err(NavigationError::NonPositiveDt);
        }

        let mut corrected_acc = linear_acceleration - self.params.bias_acc;
        corrected_acc.z -= self.params.gravity;
        let corrected_gyro = angular_velocity - self.params.bias_gyro;

        self.state.linear_velocity += corrected_acc * dt;
        self.state.translation += self.state.linear_velocity * dt;
        self.state.rotation *= UnitQuaternion::from_scaled_axis(corrected_gyro * dt);
        self.state.angular_velocity = angular_velocity;

        Ok(())
    }

    pub fn get_state(&self) -> IntegrationState {
        self.state.clone()
    }

    pub fn reset_state(
        &mut self,
        transform: Isometry3<f64>,
        linear_velocity: Vector3<f64>,
        angular_velocity: Vector3<f64>,
    ) {
        self.state.translation = transform.translation.vector;
        self.state.rotation = transform.rotation;
        self.state.linear_velocity = linear_velocity;
        self.state.angular_velocity = angular_velocity;
    }

    pub fn reset_bias(
        &mut self,
        imu_angular_velocity_bias: Vector3<f64>,
        imu_linear_acceleration_bias: Vector3<f64>,
    ) {
        self.params.bias_gyro = imu_angular_velocity_bias;
        self.params.bias_acc = imu_linear_acceleration_bias;
    }
}

#[cfg(test)]
mod tests {
    use super::{ImuIntegration, ImuParameters, NavigationError};
    use nalgebra::Vector3;

    #[test]
    fn add_imu_rejects_non_positive_dt() {
        let params = ImuParameters::default();
        let mut imu = ImuIntegration::new(&params);
        let err = imu
            .add_imu(0.0, Vector3::zeros(), Vector3::zeros())
            .unwrap_err();
        assert_eq!(err, NavigationError::NonPositiveDt);
    }

    #[test]
    fn add_imu_updates_state() {
        let params = ImuParameters::default();
        let mut imu = ImuIntegration::new(&params);

        imu.add_imu(
            0.1,
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 0.1),
        )
        .expect("valid dt");

        let state = imu.get_state();
        assert!(state.translation.x > 0.0);
        assert!(state.linear_velocity.x > 0.0);
        assert!(state.rotation.scaled_axis().z > 0.0);
    }
}

#[derive(Debug, Clone)]
pub struct ImuParameters {
    pub gravity: f64,
    pub accel_noise: f64,
    pub gyro_noise: f64,
    pub accel_bias_noise: f64,
    pub gyro_bias_noise: f64,
    pub integration_noise: f64,
    pub bias_acc: nalgebra::Vector3<f64>, // TODO if we have a known bias, it is removed while applying the calibration?
    pub bias_gyro: nalgebra::Vector3<f64>,
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
            bias_acc: nalgebra::Vector3::new(0.0, 0.0, 0.0),
            bias_gyro: nalgebra::Vector3::new(0.0, 0.0, 0.0),
        }
    }
}

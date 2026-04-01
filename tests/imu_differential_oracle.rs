#![cfg(not(feature = "pure-rust-navigation"))]
#![allow(deprecated)]

use gtsam_rs_navigation as rust_nav;
use gtsam_sys::{config::ImuParameters, ImuIntegration as OracleImuIntegration};
use nalgebra::Vector3;

fn load_measurements(limit: usize) -> Vec<(Vector3<f64>, Vector3<f64>)> {
    let data_path = format!(
        "{}/gtsam/gtsam/examples/Data/quadraped_imu_data.csv",
        env!("CARGO_MANIFEST_DIR")
    );
    let csv = std::fs::read_to_string(&data_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", data_path, e));

    csv.lines()
        .skip(1)
        .filter_map(|line| {
            let cols: Vec<&str> = line.split(',').collect();
            if cols.len() <= 31 {
                return None;
            }
            let wx = cols[17].parse::<f64>().ok()?;
            let wy = cols[18].parse::<f64>().ok()?;
            let wz = cols[19].parse::<f64>().ok()?;
            let ax = cols[29].parse::<f64>().ok()?;
            let ay = cols[30].parse::<f64>().ok()?;
            let az = cols[31].parse::<f64>().ok()?;
            Some((Vector3::new(ax, ay, az), Vector3::new(wx, wy, wz)))
        })
        .take(limit)
        .collect()
}

#[test]
fn navigation_matches_cpp_oracle_on_static_dataset_window() {
    let params = ImuParameters::default();
    let mut oracle = OracleImuIntegration::new(&params);

    let rust_params = rust_nav::ImuParameters {
        gravity: params.gravity,
        accel_noise: params.accel_noise,
        gyro_noise: params.gyro_noise,
        accel_bias_noise: params.accel_bias_noise,
        gyro_bias_noise: params.gyro_bias_noise,
        integration_noise: params.integration_noise,
        bias_acc: params.bias_acc,
        bias_gyro: params.bias_gyro,
    };
    let mut rust = rust_nav::ImuIntegration::new(&rust_params);

    let dt = 1.0 / 400.0;
    for (acc, gyro) in load_measurements(2_000) {
        oracle.add_imu(dt, acc, gyro);
        rust.add_imu(dt, acc, gyro).expect("valid dt");
    }

    let o = oracle.get_state();
    let r = rust.get_state();

    let t_diff = (o.translation - r.translation).norm();
    let v_diff = (o.linear_velocity - r.linear_velocity).norm();
    let w_diff = (o.angular_velocity - r.angular_velocity).norm();
    let rot_diff = (o.rotation.inverse() * r.rotation).scaled_axis().norm();

    assert!(t_diff < 2.0, "translation diff too large: {t_diff}");
    assert!(v_diff < 2.0, "velocity diff too large: {v_diff}");
    assert!(w_diff < 0.1, "angular velocity diff too large: {w_diff}");
    assert!(rot_diff < 0.5, "rotation diff too large: {rot_diff}");
}

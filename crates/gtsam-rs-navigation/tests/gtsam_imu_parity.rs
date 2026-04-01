use gtsam_rs_navigation::{ImuIntegration, ImuParameters};
use nalgebra::Vector3;

#[test]
fn imu_preintegration_loaded_simulation_data_smoke() {
    // Ported in spirit from gtsam/tests/testImuPreintegration.cpp:
    // load the same CSV and integrate measurements with the same dt.
    let data_path = format!(
        "{}/../../gtsam/gtsam/examples/Data/quadraped_imu_data.csv",
        env!("CARGO_MANIFEST_DIR")
    );
    let csv = std::fs::read_to_string(&data_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", data_path, e));

    let mut imu = ImuIntegration::new(&ImuParameters::default());
    let dt = 1.0 / 400.0;

    for line in csv.lines().skip(1) {
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() <= 31 {
            continue;
        }

        let wx = cols[17].parse::<f64>().unwrap_or(0.0);
        let wy = cols[18].parse::<f64>().unwrap_or(0.0);
        let wz = cols[19].parse::<f64>().unwrap_or(0.0);
        let ax = cols[29].parse::<f64>().unwrap_or(0.0);
        let ay = cols[30].parse::<f64>().unwrap_or(0.0);
        let az = cols[31].parse::<f64>().unwrap_or(0.0);

        imu.add_imu(dt, Vector3::new(ax, ay, az), Vector3::new(wx, wy, wz))
            .expect("valid dt");
    }

    let state = imu.get_state();
    assert!(state.translation.iter().all(|v| v.is_finite()));
    assert!(state.linear_velocity.iter().all(|v| v.is_finite()));
    assert!(state.rotation.scaled_axis().iter().all(|v| v.is_finite()));
    assert!(state.translation.z.abs() < 1.0);
}

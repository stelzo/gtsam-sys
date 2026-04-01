#![cfg(feature = "pure-rust-navigation")]
#![allow(deprecated)]

use gtsam_sys::{config::ImuParameters, ImuIntegration};
use nalgebra::Vector3;

#[test]
fn pure_rust_backend_updates_state() {
    let params = ImuParameters::default();
    let mut integration = ImuIntegration::new(&params);

    integration.add_imu(
        0.1,
        Vector3::new(0.5, 0.0, 0.0),
        Vector3::new(0.0, 0.0, 0.01),
    );

    let state = integration.get_state();
    assert!(state.translation.x > 0.0);
    assert!(state.linear_velocity.x > 0.0);
}

#[test]
#[should_panic(expected = "dt must be positive")]
fn pure_rust_backend_rejects_non_positive_dt() {
    let params = ImuParameters::default();
    let mut integration = ImuIntegration::new(&params);
    integration.add_imu(0.0, Vector3::zeros(), Vector3::zeros());
}

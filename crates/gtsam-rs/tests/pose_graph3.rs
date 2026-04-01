use std::collections::BTreeMap;

use gtsam_rs::geometry::{LieGroup, Pose3, Rot3};
use gtsam_rs::graph::Key;
use gtsam_rs::slam::{
    BetweenFactorPose3, GaussNewtonPoseGraph3, Noise6, PoseGraph3, PriorFactorPose3,
};
use nalgebra::Vector3;

#[test]
fn solves_pose3_graph_without_cpp_backend() {
    let mut graph = PoseGraph3::new();
    let noise = Noise6::isotropic(0.1);

    let x0 = Key(0);
    let x1 = Key(1);
    let x2 = Key(2);

    graph.add_prior(PriorFactorPose3::new(x0, Pose3::identity(), noise));
    graph.add_between(BetweenFactorPose3::new(
        x0,
        x1,
        Pose3::new(Rot3::identity(), Vector3::new(1.0, 0.0, 0.0)),
        noise,
    ));
    graph.add_between(BetweenFactorPose3::new(
        x1,
        x2,
        Pose3::new(Rot3::identity(), Vector3::new(1.0, 0.0, 0.0)),
        noise,
    ));

    let mut initial = BTreeMap::new();
    initial.insert(
        x0,
        Pose3::new(
            Rot3::from_scaled_axis(Vector3::new(0.01, -0.01, 0.01)),
            Vector3::new(0.05, -0.05, 0.0),
        ),
    );
    initial.insert(
        x1,
        Pose3::new(
            Rot3::from_scaled_axis(Vector3::new(0.2, 0.1, -0.1)),
            Vector3::new(0.7, 0.4, -0.2),
        ),
    );
    initial.insert(
        x2,
        Pose3::new(
            Rot3::from_scaled_axis(Vector3::new(-0.2, 0.2, 0.1)),
            Vector3::new(2.4, -0.3, 0.2),
        ),
    );

    let result = GaussNewtonPoseGraph3::default()
        .optimize(&graph, &initial)
        .expect("optimizer should succeed");

    assert!(result.summary.final_error < result.summary.initial_error);

    let x2_est = result.values.get(&x2).expect("x2");
    let expected = Pose3::new(Rot3::identity(), Vector3::new(2.0, 0.0, 0.0));
    let err = expected.inverse().compose(x2_est);
    assert!(err.translation.norm() < 1e-2);
    assert!(err.rotation.scaled_axis().norm() < 1e-2);
}

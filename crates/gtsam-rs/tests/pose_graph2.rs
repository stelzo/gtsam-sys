use std::collections::BTreeMap;

use gtsam_rs::geometry::{LieGroup, Pose2, Rot2};
use gtsam_rs::graph::Key;
use gtsam_rs::slam::{BetweenFactorPose2, GaussNewtonPoseGraph2, Noise3, PoseGraph2, PriorFactorPose2};
use nalgebra::Vector2;

#[test]
fn solves_pose_graph_without_cpp_backend() {
    let mut graph = PoseGraph2::new();
    let noise = Noise3::isotropic(0.1);

    let x0 = Key(0);
    let x1 = Key(1);
    let x2 = Key(2);

    graph.add_prior(PriorFactorPose2::new(x0, Pose2::identity(), noise));
    graph.add_between(BetweenFactorPose2::new(
        x0,
        x1,
        Pose2::new(Rot2::from_angle(0.0), Vector2::new(1.0, 0.0)),
        noise,
    ));
    graph.add_between(BetweenFactorPose2::new(
        x1,
        x2,
        Pose2::new(Rot2::from_angle(0.0), Vector2::new(1.0, 0.0)),
        noise,
    ));

    let mut initial = BTreeMap::new();
    initial.insert(x0, Pose2::new(Rot2::from_angle(0.0), Vector2::new(0.0, 0.0)));
    initial.insert(x1, Pose2::new(Rot2::from_angle(0.2), Vector2::new(0.7, 0.4)));
    initial.insert(x2, Pose2::new(Rot2::from_angle(-0.2), Vector2::new(2.4, -0.3)));

    let result = GaussNewtonPoseGraph2::default()
        .optimize(&graph, &initial)
        .expect("optimizer should succeed");

    assert!(result.summary.final_error < result.summary.initial_error);

    let x2_est = result.values.get(&x2).expect("x2");
    let expected_x2 = Pose2::new(Rot2::from_angle(0.0), Vector2::new(2.0, 0.0));
    let err = expected_x2.inverse().compose(x2_est);
    assert!(err.translation.norm() < 1e-2);
    assert!(err.rotation.angle().abs() < 1e-2);
}

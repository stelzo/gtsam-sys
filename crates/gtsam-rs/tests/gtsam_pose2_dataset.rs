use std::collections::BTreeMap;

use gtsam_rs::geometry::{LocalCoordinates, Pose2, Rot2};
use gtsam_rs::graph::Key;
use gtsam_rs::slam::{BetweenFactorPose2, GaussNewtonPoseGraph2, Noise3, PoseGraph2, PriorFactorPose2};
use nalgebra::Vector2;

fn parse_vertices(path: &str) -> BTreeMap<Key, Pose2> {
    let txt = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {}: {}", path, e));
    let mut out = BTreeMap::new();
    for line in txt.lines() {
        if !line.starts_with("VERTEX_SE2 ") {
            continue;
        }
        let c: Vec<&str> = line.split_whitespace().collect();
        assert!(c.len() >= 5, "bad vertex line: {}", line);
        let k = c[1].parse::<u64>().expect("key");
        let x = c[2].parse::<f64>().expect("x");
        let y = c[3].parse::<f64>().expect("y");
        let t = c[4].parse::<f64>().expect("theta");
        out.insert(
            Key(k),
            Pose2::new(Rot2::from_angle(t), Vector2::new(x, y)),
        );
    }
    out
}

fn parse_edges(path: &str) -> Vec<(Key, Key, Pose2)> {
    let txt = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {}: {}", path, e));
    let mut out = Vec::new();
    for line in txt.lines() {
        if !line.starts_with("EDGE_SE2 ") {
            continue;
        }
        let c: Vec<&str> = line.split_whitespace().collect();
        assert!(c.len() >= 6, "bad edge line: {}", line);
        let i = Key(c[1].parse::<u64>().expect("i"));
        let j = Key(c[2].parse::<u64>().expect("j"));
        let dx = c[3].parse::<f64>().expect("dx");
        let dy = c[4].parse::<f64>().expect("dy");
        let dt = c[5].parse::<f64>().expect("dtheta");
        out.push((i, j, Pose2::new(Rot2::from_angle(dt), Vector2::new(dx, dy))));
    }
    out
}

#[test]
fn noisy_toy_graph_matches_gtsam_optimized_reference() {
    let base = format!("{}/../../gtsam/gtsam/examples/Data", env!("CARGO_MANIFEST_DIR"));
    let noisy_path = format!("{}/noisyToyGraph.txt", base);
    let optimized_path = format!("{}/optimizedNoisyToyGraph.txt", base);

    let initial = parse_vertices(&noisy_path);
    let expected = parse_vertices(&optimized_path);
    let edges = parse_edges(&noisy_path);

    let noise = Noise3::isotropic(1.0);
    let mut graph = PoseGraph2::new();
    let anchor = initial
        .iter()
        .next()
        .map(|(k, p)| (*k, *p))
        .expect("non-empty graph");
    graph.add_prior(PriorFactorPose2::new(anchor.0, anchor.1, Noise3::isotropic(1e-3)));
    for (i, j, z) in edges {
        graph.add_between(BetweenFactorPose2::new(i, j, z, noise));
    }

    let solver = GaussNewtonPoseGraph2::default();
    let result = solver.optimize(&graph, &initial).expect("optimize");
    assert!(result.summary.converged, "optimizer did not converge");

    for (k, p_exp) in expected {
        let p_est = result.values.get(&k).unwrap_or_else(|| panic!("missing key {}", k.0));
        let err = p_exp.local_coordinates(p_est);
        assert!(
            err.norm() < 5e-2,
            "key {} mismatch: local error norm {}",
            k.0,
            err.norm()
        );
    }
}

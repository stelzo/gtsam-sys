use std::collections::BTreeMap;
use std::time::Instant;

use gtsam_rs::geometry::{Pose2, Rot2};
use gtsam_rs::graph::Key;
use gtsam_rs::slam::{BetweenFactorPose2, GaussNewtonPoseGraph2, Noise3, PoseGraph2, PriorFactorPose2};
use nalgebra::Vector2;

fn load_g2o_pose2(path: &str) -> Result<(PoseGraph2, BTreeMap<Key, Pose2>), String> {
    let txt = std::fs::read_to_string(path).map_err(|e| format!("failed to read {}: {}", path, e))?;
    let mut graph = PoseGraph2::new();
    let mut initial = BTreeMap::<Key, Pose2>::new();

    for line in txt.lines() {
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.is_empty() {
            continue;
        }
        match cols[0] {
            "VERTEX_SE2" => {
                if cols.len() < 5 {
                    return Err(format!("bad VERTEX_SE2 line: {}", line));
                }
                let id = cols[1].parse::<u64>().map_err(|e| format!("bad vertex id: {}", e))?;
                let x = cols[2].parse::<f64>().map_err(|e| format!("bad x: {}", e))?;
                let y = cols[3].parse::<f64>().map_err(|e| format!("bad y: {}", e))?;
                let t = cols[4].parse::<f64>().map_err(|e| format!("bad theta: {}", e))?;
                initial.insert(Key(id), Pose2::new(Rot2::from_angle(t), Vector2::new(x, y)));
            }
            "EDGE_SE2" => {
                if cols.len() < 6 {
                    return Err(format!("bad EDGE_SE2 line: {}", line));
                }
                let i = cols[1].parse::<u64>().map_err(|e| format!("bad edge i: {}", e))?;
                let j = cols[2].parse::<u64>().map_err(|e| format!("bad edge j: {}", e))?;
                let dx = cols[3].parse::<f64>().map_err(|e| format!("bad dx: {}", e))?;
                let dy = cols[4].parse::<f64>().map_err(|e| format!("bad dy: {}", e))?;
                let dt = cols[5].parse::<f64>().map_err(|e| format!("bad dtheta: {}", e))?;
                graph.add_between(BetweenFactorPose2::new(
                    Key(i),
                    Key(j),
                    Pose2::new(Rot2::from_angle(dt), Vector2::new(dx, dy)),
                    Noise3::isotropic(1.0),
                ));
            }
            _ => {}
        }
    }

    let first = initial
        .iter()
        .next()
        .map(|(k, p)| (*k, *p))
        .ok_or_else(|| "no VERTEX_SE2 entries found".to_string())?;
    graph.add_prior(PriorFactorPose2::new(first.0, first.1, Noise3::isotropic(1e-3)));
    Ok((graph, initial))
}

fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| "usage: cargo run -p gtsam-rs --release --example pose2_g2o -- <file.g2o>".to_string())?;

    let (graph, initial) = load_g2o_pose2(&path)?;
    println!(
        "Loaded {} poses, {} between factors",
        initial.len(),
        graph.between.len()
    );

    let solver = GaussNewtonPoseGraph2 {
        max_iterations: 200,
        step_tolerance: 1e-4,
        ..Default::default()
    };
    let start = Instant::now();
    let result = solver.optimize(&graph, &initial)?;
    let elapsed = start.elapsed();
    println!(
        "Converged={} iterations={} final_error={:.6e} duration={:?}",
        result.summary.converged, result.summary.iterations, result.summary.final_error, elapsed
    );
    Ok(())
}

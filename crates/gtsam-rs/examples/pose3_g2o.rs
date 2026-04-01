use std::collections::BTreeMap;
use std::time::Instant;

use gtsam_rs::geometry::{Pose3, Rot3};
use gtsam_rs::graph::Key;
use gtsam_rs::slam::{BetweenFactorPose3, GaussNewtonPoseGraph3, Noise6, PoseGraph3, PriorFactorPose3};
use nalgebra::{Quaternion, SMatrix, UnitQuaternion, Vector3};

fn noise6_from_g2o_information(cols: &[&str]) -> Noise6 {
    // g2o stores upper-triangular information matrix entries after pose measurement.
    // Diagonal indices in the 21-entry upper-tri layout:
    // (0,0)=0, (1,1)=6, (2,2)=11, (3,3)=15, (4,4)=18, (5,5)=20
    let info_start = 10usize;
    let diag_idx = [0usize, 6, 11, 15, 18, 20];
    let mut d = [1.0f64; 6];
    for (k, idx) in diag_idx.iter().enumerate() {
        let col = info_start + idx;
        if col < cols.len() {
            let val = cols[col].parse::<f64>().unwrap_or(1.0).abs();
            d[k] = if val > 1e-12 { val } else { 1.0 };
        }
    }
    Noise6 {
        sigma_x: 1.0 / d[0].sqrt(),
        sigma_y: 1.0 / d[1].sqrt(),
        sigma_z: 1.0 / d[2].sqrt(),
        sigma_roll: 1.0 / d[3].sqrt(),
        sigma_pitch: 1.0 / d[4].sqrt(),
        sigma_yaw: 1.0 / d[5].sqrt(),
    }
}

fn sqrt_info6_from_g2o_information(cols: &[&str]) -> Option<SMatrix<f64, 6, 6>> {
    let info_start = 10usize;
    if cols.len() < info_start + 21 {
        return None;
    }
    let mut info = SMatrix::<f64, 6, 6>::zeros();
    let mut idx = 0usize;
    for r in 0..6 {
        for c in r..6 {
            let v = cols[info_start + idx].parse::<f64>().ok()?;
            info[(r, c)] = v;
            info[(c, r)] = v;
            idx += 1;
        }
    }
    nalgebra::Cholesky::new(info).map(|c| c.l().transpose())
}

fn load_g2o_pose3(path: &str) -> Result<(PoseGraph3, BTreeMap<Key, Pose3>), String> {
    let txt = std::fs::read_to_string(path).map_err(|e| format!("failed to read {}: {}", path, e))?;
    let mut graph = PoseGraph3::new();
    let mut initial = BTreeMap::<Key, Pose3>::new();

    for line in txt.lines() {
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.is_empty() {
            continue;
        }
        match cols[0] {
            "VERTEX_SE3:QUAT" => {
                if cols.len() < 9 {
                    return Err(format!("bad VERTEX_SE3:QUAT line: {}", line));
                }
                let id = cols[1].parse::<u64>().map_err(|e| format!("bad vertex id: {}", e))?;
                let x = cols[2].parse::<f64>().map_err(|e| format!("bad x: {}", e))?;
                let y = cols[3].parse::<f64>().map_err(|e| format!("bad y: {}", e))?;
                let z = cols[4].parse::<f64>().map_err(|e| format!("bad z: {}", e))?;
                let qx = cols[5].parse::<f64>().map_err(|e| format!("bad qx: {}", e))?;
                let qy = cols[6].parse::<f64>().map_err(|e| format!("bad qy: {}", e))?;
                let qz = cols[7].parse::<f64>().map_err(|e| format!("bad qz: {}", e))?;
                let qw = cols[8].parse::<f64>().map_err(|e| format!("bad qw: {}", e))?;
                let q = UnitQuaternion::new_normalize(Quaternion::new(qw, qx, qy, qz));
                initial.insert(
                    Key(id),
                    Pose3::new(Rot3(q), Vector3::new(x, y, z)),
                );
            }
            "EDGE_SE3:QUAT" => {
                if cols.len() < 10 {
                    return Err(format!("bad EDGE_SE3:QUAT line: {}", line));
                }
                let i = cols[1].parse::<u64>().map_err(|e| format!("bad edge i: {}", e))?;
                let j = cols[2].parse::<u64>().map_err(|e| format!("bad edge j: {}", e))?;
                let dx = cols[3].parse::<f64>().map_err(|e| format!("bad dx: {}", e))?;
                let dy = cols[4].parse::<f64>().map_err(|e| format!("bad dy: {}", e))?;
                let dz = cols[5].parse::<f64>().map_err(|e| format!("bad dz: {}", e))?;
                let qx = cols[6].parse::<f64>().map_err(|e| format!("bad qx: {}", e))?;
                let qy = cols[7].parse::<f64>().map_err(|e| format!("bad qy: {}", e))?;
                let qz = cols[8].parse::<f64>().map_err(|e| format!("bad qz: {}", e))?;
                let qw = cols[9].parse::<f64>().map_err(|e| format!("bad qw: {}", e))?;
                let q = UnitQuaternion::new_normalize(Quaternion::new(qw, qx, qy, qz));
                let noise = noise6_from_g2o_information(&cols);
                let measured = Pose3::new(Rot3(q), Vector3::new(dx, dy, dz));
                if let Some(sqrt_info) = sqrt_info6_from_g2o_information(&cols) {
                    graph.add_between(BetweenFactorPose3::new_with_sqrt_info(
                        Key(i),
                        Key(j),
                        measured,
                        noise,
                        sqrt_info,
                    ));
                } else {
                    graph.add_between(BetweenFactorPose3::new(Key(i), Key(j), measured, noise));
                }
            }
            _ => {}
        }
    }

    let first = initial
        .iter()
        .next()
        .map(|(k, p)| (*k, *p))
        .ok_or_else(|| "no VERTEX_SE3:QUAT entries found".to_string())?;
    graph.add_prior(PriorFactorPose3::new(first.0, first.1, Noise6::isotropic(1e-3)));
    Ok((graph, initial))
}

fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| "usage: cargo run -p gtsam-rs --release --example pose3_g2o -- <file.g2o>".to_string())?;

    let (graph, initial) = load_g2o_pose3(&path)?;
    println!(
        "Loaded {} poses, {} between factors",
        initial.len(),
        graph.between.len()
    );

    let solver = GaussNewtonPoseGraph3 {
        max_iterations: 200,
        step_tolerance: 1e-4,
        lambda: 1e-3,
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

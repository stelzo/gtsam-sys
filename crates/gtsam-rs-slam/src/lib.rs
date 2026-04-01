#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap};

use gtsam_rs_inference::Key;
use gtsam_rs_linear::{
    compute_ordering_from_edges, symbolic_from_edges, OrderingStrategy, SparseBlockSystem3,
    SparseBlockSystem6,
};
use gtsam_rs_math::{LieGroup, Pose2, Pose3, Retract, Rot2};
use nalgebra::{DVector, Matrix3, SMatrix, SVector, Vector2, Vector3};

pub type Pose3Error = SVector<f64, 6>;
type Vec6 = SVector<f64, 6>;
type Mat6 = SMatrix<f64, 6, 6>;

#[derive(Clone, Debug, Default)]
pub struct PriorFactor;

impl PriorFactor {
    pub fn new() -> Self {
        Self
    }
}

#[derive(Clone, Debug, Default)]
pub struct BetweenFactor;

impl BetweenFactor {
    pub fn new() -> Self {
        Self
    }
}

#[derive(Clone, Debug, Default)]
pub struct BearingRangeFactor;

#[derive(Clone, Debug, Default)]
pub struct GenericProjectionFactor;

#[derive(Clone, Copy, Debug)]
pub struct Noise3 {
    pub sigma_x: f64,
    pub sigma_y: f64,
    pub sigma_theta: f64,
}

impl Noise3 {
    pub fn isotropic(sigma: f64) -> Self {
        Self {
            sigma_x: sigma,
            sigma_y: sigma,
            sigma_theta: sigma,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PriorFactorPose2 {
    pub key: Key,
    pub prior: Pose2,
    pub noise: Noise3,
}

impl PriorFactorPose2 {
    pub fn new(key: Key, prior: Pose2, noise: Noise3) -> Self {
        Self { key, prior, noise }
    }

    pub fn error(&self, estimate: &Pose2) -> Vector3<f64> {
        prior_residual(self, estimate)
    }
}

#[derive(Clone, Debug)]
pub struct BetweenFactorPose2 {
    pub key1: Key,
    pub key2: Key,
    pub measured: Pose2,
    pub noise: Noise3,
}

impl BetweenFactorPose2 {
    pub fn new(key1: Key, key2: Key, measured: Pose2, noise: Noise3) -> Self {
        Self {
            key1,
            key2,
            measured,
            noise,
        }
    }

    pub fn error(&self, p1: &Pose2, p2: &Pose2) -> Vector3<f64> {
        between_residual(self, p1, p2)
    }
}

#[derive(Clone, Debug, Default)]
pub struct PoseGraph2 {
    pub priors: Vec<PriorFactorPose2>,
    pub between: Vec<BetweenFactorPose2>,
}

impl PoseGraph2 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_prior(&mut self, factor: PriorFactorPose2) {
        self.priors.push(factor);
    }

    pub fn add_between(&mut self, factor: BetweenFactorPose2) {
        self.between.push(factor);
    }

    pub fn keys(&self) -> Vec<Key> {
        let mut keys = BTreeSet::new();
        for p in &self.priors {
            keys.insert(p.key);
        }
        for b in &self.between {
            keys.insert(b.key1);
            keys.insert(b.key2);
        }
        keys.into_iter().collect()
    }
}

#[derive(Clone, Debug)]
pub struct PoseGraph2Summary {
    pub iterations: usize,
    pub initial_error: f64,
    pub final_error: f64,
    pub converged: bool,
}

#[derive(Clone, Debug)]
pub struct PoseGraph2Result {
    pub values: BTreeMap<Key, Pose2>,
    pub summary: PoseGraph2Summary,
}

#[derive(Clone, Copy, Debug)]
pub struct Noise6 {
    pub sigma_x: f64,
    pub sigma_y: f64,
    pub sigma_z: f64,
    pub sigma_roll: f64,
    pub sigma_pitch: f64,
    pub sigma_yaw: f64,
}

impl Noise6 {
    pub fn isotropic(sigma: f64) -> Self {
        Self {
            sigma_x: sigma,
            sigma_y: sigma,
            sigma_z: sigma,
            sigma_roll: sigma,
            sigma_pitch: sigma,
            sigma_yaw: sigma,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PriorFactorPose3 {
    pub key: Key,
    pub prior: Pose3,
    pub noise: Noise6,
}

impl PriorFactorPose3 {
    pub fn new(key: Key, prior: Pose3, noise: Noise6) -> Self {
        Self { key, prior, noise }
    }

    pub fn error(&self, estimate: &Pose3) -> Pose3Error {
        pose3_residual(self.prior.inverse().compose(estimate), self.noise)
    }
}

#[derive(Clone, Debug)]
pub struct BetweenFactorPose3 {
    pub key1: Key,
    pub key2: Key,
    pub measured: Pose3,
    pub noise: Noise6,
    pub sqrt_info: Option<Mat6>,
}

impl BetweenFactorPose3 {
    pub fn new(key1: Key, key2: Key, measured: Pose3, noise: Noise6) -> Self {
        Self {
            key1,
            key2,
            measured,
            noise,
            sqrt_info: None,
        }
    }

    pub fn new_with_sqrt_info(
        key1: Key,
        key2: Key,
        measured: Pose3,
        noise: Noise6,
        sqrt_info: Mat6,
    ) -> Self {
        Self {
            key1,
            key2,
            measured,
            noise,
            sqrt_info: Some(sqrt_info),
        }
    }

    pub fn error(&self, p1: &Pose3, p2: &Pose3) -> Pose3Error {
        between3_residual(self, p1, p2)
    }
}

#[derive(Clone, Debug, Default)]
pub struct PoseGraph3 {
    pub priors: Vec<PriorFactorPose3>,
    pub between: Vec<BetweenFactorPose3>,
}

impl PoseGraph3 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_prior(&mut self, factor: PriorFactorPose3) {
        self.priors.push(factor);
    }

    pub fn add_between(&mut self, factor: BetweenFactorPose3) {
        self.between.push(factor);
    }

    pub fn keys(&self) -> Vec<Key> {
        let mut keys = BTreeSet::new();
        for p in &self.priors {
            keys.insert(p.key);
        }
        for b in &self.between {
            keys.insert(b.key1);
            keys.insert(b.key2);
        }
        keys.into_iter().collect()
    }
}

#[derive(Clone, Debug)]
pub struct PoseGraph3Summary {
    pub iterations: usize,
    pub initial_error: f64,
    pub final_error: f64,
    pub converged: bool,
}

#[derive(Clone, Debug)]
pub struct PoseGraph3Result {
    pub values: BTreeMap<Key, Pose3>,
    pub summary: PoseGraph3Summary,
}

#[derive(Clone, Debug)]
pub struct GaussNewtonPoseGraph2 {
    pub max_iterations: usize,
    pub step_tolerance: f64,
    pub lambda: f64,
    pub ordering: OrderingStrategy,
}

impl Default for GaussNewtonPoseGraph2 {
    fn default() -> Self {
        Self {
            max_iterations: 40,
            step_tolerance: 1e-9,
            lambda: 1e-9,
            ordering: OrderingStrategy::MinDegree,
        }
    }
}

impl GaussNewtonPoseGraph2 {
    pub fn optimize(
        &self,
        graph: &PoseGraph2,
        initial: &BTreeMap<Key, Pose2>,
    ) -> Result<PoseGraph2Result, String> {
        if graph.priors.is_empty() {
            return Err("pose graph requires at least one prior factor".to_string());
        }

        let keys = graph.keys();
        if keys.is_empty() {
            return Err("pose graph is empty".to_string());
        }
        for key in &keys {
            if !initial.contains_key(key) {
                return Err(format!("missing initial value for key {}", key.0));
            }
        }

        let anchor = graph.priors[0].key;
        let anchor_pose = graph.priors[0].prior;
        let variable_keys: Vec<Key> = keys.into_iter().filter(|k| *k != anchor).collect();
        let key_to_var: BTreeMap<Key, usize> = variable_keys
            .iter()
            .enumerate()
            .map(|(i, k)| (*k, i))
            .collect();
        let edges = variable_edges(graph, &key_to_var);
        let ordering = compute_ordering_from_edges(key_to_var.len(), &edges, self.ordering);
        let symbolic = symbolic_from_edges(key_to_var.len(), &edges, &ordering)?;

        let all_keys = graph.keys();
        let key_to_slot: HashMap<Key, usize> = all_keys
            .iter()
            .enumerate()
            .map(|(i, k)| (*k, i))
            .collect();
        let variable_slots: Vec<usize> = variable_keys
            .iter()
            .map(|k| {
                key_to_slot
                    .get(k)
                    .copied()
                    .ok_or_else(|| format!("missing slot for key {}", k.0))
            })
            .collect::<Result<_, _>>()?;

        let mut poses: Vec<Pose2> = all_keys
            .iter()
            .map(|k| {
                initial
                    .get(k)
                    .copied()
                    .ok_or_else(|| format!("missing initial value for key {}", k.0))
            })
            .collect::<Result<_, _>>()?;
        if let Some(&anchor_slot) = key_to_slot.get(&anchor) {
            poses[anchor_slot] = anchor_pose;
        }

        let mut prior_indexed = Vec::with_capacity(graph.priors.len());
        for f in &graph.priors {
            let slot = *key_to_slot
                .get(&f.key)
                .ok_or_else(|| format!("missing slot for prior key {}", f.key.0))?;
            let var = key_to_var.get(&f.key).copied();
            prior_indexed.push(PriorFactorPose2Indexed {
                slot,
                var,
                prior: f.prior,
                noise: f.noise,
            });
        }

        let mut between_indexed = Vec::with_capacity(graph.between.len());
        for f in &graph.between {
            let slot1 = *key_to_slot
                .get(&f.key1)
                .ok_or_else(|| format!("missing slot for key {}", f.key1.0))?;
            let slot2 = *key_to_slot
                .get(&f.key2)
                .ok_or_else(|| format!("missing slot for key {}", f.key2.0))?;
            let var1 = key_to_var.get(&f.key1).copied();
            let var2 = key_to_var.get(&f.key2).copied();
            between_indexed.push(BetweenFactorPose2Indexed {
                slot1,
                slot2,
                var1,
                var2,
                measured: f.measured,
                noise: f.noise,
            });
        }

        let mut state = flatten_state(initial, &variable_keys)?;
        write_pose2_state(&state, &variable_slots, &mut poses);
        let initial_error = total_error_indexed(&prior_indexed, &between_indexed, &poses);

        let mut final_error = initial_error;
        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..self.max_iterations {
            iterations = iter + 1;
            write_pose2_state(&state, &variable_slots, &mut poses);
            let (system, err) =
                linearize_system_indexed(variable_keys.len(), &prior_indexed, &between_indexed, &poses);
            final_error = err;

            let delta = system.solve_with_symbolic(&symbolic, self.lambda)?;
            state += &delta;

            if delta.norm() < self.step_tolerance {
                converged = true;
                break;
            }
        }

        let values = compose_values(initial, &variable_keys, &state, anchor, anchor_pose)?;

        Ok(PoseGraph2Result {
            values,
            summary: PoseGraph2Summary {
                iterations,
                initial_error,
                final_error,
                converged,
            },
        })
    }
}

#[derive(Clone, Copy)]
struct PriorFactorPose2Indexed {
    slot: usize,
    var: Option<usize>,
    prior: Pose2,
    noise: Noise3,
}

#[derive(Clone, Copy)]
struct BetweenFactorPose2Indexed {
    slot1: usize,
    slot2: usize,
    var1: Option<usize>,
    var2: Option<usize>,
    measured: Pose2,
    noise: Noise3,
}

#[derive(Clone, Debug)]
pub struct GaussNewtonPoseGraph3 {
    pub max_iterations: usize,
    pub step_tolerance: f64,
    pub lambda: f64,
    pub ordering: OrderingStrategy,
}

impl Default for GaussNewtonPoseGraph3 {
    fn default() -> Self {
        Self {
            max_iterations: 40,
            step_tolerance: 1e-9,
            lambda: 1e-9,
            ordering: OrderingStrategy::MinDegree,
        }
    }
}

impl GaussNewtonPoseGraph3 {
    pub fn optimize(
        &self,
        graph: &PoseGraph3,
        initial: &BTreeMap<Key, Pose3>,
    ) -> Result<PoseGraph3Result, String> {
        if graph.priors.is_empty() {
            return Err("pose graph requires at least one prior factor".to_string());
        }

        let keys = graph.keys();
        if keys.is_empty() {
            return Err("pose graph is empty".to_string());
        }
        for key in &keys {
            if !initial.contains_key(key) {
                return Err(format!("missing initial value for key {}", key.0));
            }
        }

        let anchor = graph.priors[0].key;
        let anchor_pose = graph.priors[0].prior;
        let variable_keys: Vec<Key> = keys.into_iter().filter(|k| *k != anchor).collect();
        let key_to_var: BTreeMap<Key, usize> = variable_keys
            .iter()
            .enumerate()
            .map(|(i, k)| (*k, i))
            .collect();
        let edges = variable_edges3(graph, &key_to_var);
        let ordering = compute_ordering_from_edges(key_to_var.len(), &edges, self.ordering);
        let symbolic = symbolic_from_edges(key_to_var.len(), &edges, &ordering)?;

        let mut values = initial.clone();
        values.insert(anchor, anchor_pose);
        let initial_error = total_error3(graph, &values);

        let mut final_error = initial_error;
        let mut converged = false;
        let mut iterations = 0;
        let mut lambda = self.lambda;

        for iter in 0..self.max_iterations {
            iterations = iter + 1;
            values.insert(anchor, anchor_pose);
            let (system, err) = linearize_system3(graph, &values, &key_to_var);
            final_error = err;

            let delta = system.solve_with_symbolic(&symbolic, lambda)?;

            let mut accepted = false;
            let mut step = 1.0f64;
            let mut accepted_delta_norm = 0.0f64;
            let mut candidate_values = values.clone();
            for _ in 0..8 {
                let mut previous_values = Vec::with_capacity(variable_keys.len());
                for (i, key) in variable_keys.iter().enumerate() {
                    let mut dv = Vec6::zeros();
                    for r in 0..6 {
                        dv[r] = step * delta[i * 6 + r];
                    }
                    let current = candidate_values
                        .get(key)
                        .copied()
                        .ok_or_else(|| format!("missing value for key {}", key.0))?;
                    let p = candidate_values
                        .insert(*key, current.retract(&dv))
                        .ok_or_else(|| format!("missing value for key {}", key.0))?;
                    previous_values.push((*key, p));
                }
                candidate_values.insert(anchor, anchor_pose);
                let candidate_err = total_error3(graph, &candidate_values);
                if candidate_err.is_finite() && candidate_err <= err {
                    values = candidate_values;
                    final_error = candidate_err;
                    accepted = true;
                    accepted_delta_norm = (step * &delta).norm();
                    lambda = (lambda * 0.5).max(1e-9);
                    break;
                }
                for (key, old_value) in previous_values.into_iter().rev() {
                    candidate_values.insert(key, old_value);
                }
                candidate_values.insert(anchor, anchor_pose);
                step *= 0.5;
            }

            if !accepted {
                lambda = (lambda * 10.0).min(1e3);
                continue;
            }

            if accepted_delta_norm < self.step_tolerance {
                converged = true;
                break;
            }
        }

        Ok(PoseGraph3Result {
            values,
            summary: PoseGraph3Summary {
                iterations,
                initial_error,
                final_error,
                converged,
            },
        })
    }
}

fn flatten_state(values: &BTreeMap<Key, Pose2>, variable_keys: &[Key]) -> Result<DVector<f64>, String> {
    let mut x = DVector::<f64>::zeros(variable_keys.len() * 3);
    for (i, key) in variable_keys.iter().enumerate() {
        let pose = values
            .get(key)
            .ok_or_else(|| format!("missing initial value for key {}", key.0))?;
        x[i * 3] = pose.translation.x;
        x[i * 3 + 1] = pose.translation.y;
        x[i * 3 + 2] = pose.rotation.angle();
    }
    Ok(x)
}

fn compose_values(
    initial: &BTreeMap<Key, Pose2>,
    variable_keys: &[Key],
    x: &DVector<f64>,
    anchor: Key,
    anchor_pose: Pose2,
) -> Result<BTreeMap<Key, Pose2>, String> {
    let mut values = initial.clone();
    for (i, key) in variable_keys.iter().enumerate() {
        let pose = Pose2::new(
            Rot2::from_angle(x[i * 3 + 2]),
            Vector2::new(x[i * 3], x[i * 3 + 1]),
        );
        values.insert(*key, pose);
    }
    values.insert(anchor, anchor_pose);
    Ok(values)
}

fn write_pose2_state(state: &DVector<f64>, variable_slots: &[usize], poses: &mut [Pose2]) {
    for (i, &slot) in variable_slots.iter().enumerate() {
        poses[slot] = Pose2::new(
            Rot2::from_angle(state[i * 3 + 2]),
            Vector2::new(state[i * 3], state[i * 3 + 1]),
        );
    }
}

fn pose3_residual(error_pose: Pose3, noise: Noise6) -> Vec6 {
    let rv = error_pose.rotation.scaled_axis();
    Vec6::new(
        error_pose.translation.x / noise.sigma_x,
        error_pose.translation.y / noise.sigma_y,
        error_pose.translation.z / noise.sigma_z,
        rv.x / noise.sigma_roll,
        rv.y / noise.sigma_pitch,
        rv.z / noise.sigma_yaw,
    )
}

fn pose3_residual_raw(error_pose: Pose3) -> Vec6 {
    let rv = error_pose.rotation.scaled_axis();
    Vec6::new(
        error_pose.translation.x,
        error_pose.translation.y,
        error_pose.translation.z,
        rv.x,
        rv.y,
        rv.z,
    )
}

fn between3_residual(f: &BetweenFactorPose3, p1: &Pose3, p2: &Pose3) -> Vec6 {
    let predicted = p1.inverse().compose(p2);
    let error_pose = f.measured.inverse().compose(&predicted);
    if let Some(w) = &f.sqrt_info {
        *w * pose3_residual_raw(error_pose)
    } else {
        pose3_residual(error_pose, f.noise)
    }
}

fn prior3_residual_jacobian_numeric(f: &PriorFactorPose3, estimate: &Pose3) -> (Vec6, Mat6) {
    let r0 = pose3_residual(f.prior.inverse().compose(estimate), f.noise);
    let mut j = Mat6::zeros();
    let eps = 1e-6;
    for c in 0..6 {
        let mut dv = Vec6::zeros();
        dv[c] = eps;
        let p = estimate.retract(&dv);
        let re = pose3_residual(f.prior.inverse().compose(&p), f.noise);
        j.set_column(c, &((re - r0) / eps));
    }
    (r0, j)
}

fn between3_residual_jacobians_numeric(
    f: &BetweenFactorPose3,
    p1: &Pose3,
    p2: &Pose3,
) -> (Vec6, Mat6, Mat6) {
    let r0 = between3_residual(f, p1, p2);
    let mut j1 = Mat6::zeros();
    let mut j2 = Mat6::zeros();
    let eps = 1e-6;
    for c in 0..6 {
        let mut dv = Vec6::zeros();
        dv[c] = eps;
        let p1p = p1.retract(&dv);
        let r1 = between3_residual(f, &p1p, p2);
        j1.set_column(c, &((r1 - r0) / eps));

        let p2p = p2.retract(&dv);
        let r2 = between3_residual(f, p1, &p2p);
        j2.set_column(c, &((r2 - r0) / eps));
    }
    (r0, j1, j2)
}

fn wrap_angle(theta: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut a = (theta + std::f64::consts::PI) % two_pi;
    if a < 0.0 {
        a += two_pi;
    }
    a - std::f64::consts::PI
}

fn pose2_components(p: &Pose2) -> (f64, f64, f64, f64, f64) {
    let x = p.translation.x;
    let y = p.translation.y;
    let t = p.rotation.angle();
    (x, y, t, t.cos(), t.sin())
}

fn prior_residual(f: &PriorFactorPose2, estimate: &Pose2) -> Vector3<f64> {
    let (x, y, t, _, _) = pose2_components(estimate);
    let (px, py, pt, cp, sp) = pose2_components(&f.prior);

    let dx = x - px;
    let dy = y - py;
    Vector3::new(
        (cp * dx + sp * dy) / f.noise.sigma_x,
        (-sp * dx + cp * dy) / f.noise.sigma_y,
        wrap_angle(t - pt) / f.noise.sigma_theta,
    )
}

fn prior_residual_jacobian(
    f: &PriorFactorPose2,
    estimate: &Pose2,
) -> (Vector3<f64>, Matrix3<f64>) {
    let (x, y, t, _, _) = pose2_components(estimate);
    let (px, py, pt, cp, sp) = pose2_components(&f.prior);

    let dx = x - px;
    let dy = y - py;
    let r0 = (cp * dx + sp * dy) / f.noise.sigma_x;
    let r1 = (-sp * dx + cp * dy) / f.noise.sigma_y;
    let r2 = wrap_angle(t - pt) / f.noise.sigma_theta;

    let mut j = Matrix3::<f64>::zeros();
    j[(0, 0)] = cp / f.noise.sigma_x;
    j[(0, 1)] = sp / f.noise.sigma_x;
    j[(1, 0)] = -sp / f.noise.sigma_y;
    j[(1, 1)] = cp / f.noise.sigma_y;
    j[(2, 2)] = 1.0 / f.noise.sigma_theta;

    (Vector3::new(r0, r1, r2), j)
}

fn between_residual(f: &BetweenFactorPose2, p1: &Pose2, p2: &Pose2) -> Vector3<f64> {
    let (x1, y1, t1, c1, s1) = pose2_components(p1);
    let (x2, y2, t2, _, _) = pose2_components(p2);
    let (mx, my, mt, cm, sm) = pose2_components(&f.measured);

    let dx = x2 - x1;
    let dy = y2 - y1;
    let tx = c1 * dx + s1 * dy;
    let ty = -s1 * dx + c1 * dy;
    let dtheta = wrap_angle(t2 - t1);

    let etx = tx - mx;
    let ety = ty - my;
    Vector3::new(
        (cm * etx + sm * ety) / f.noise.sigma_x,
        (-sm * etx + cm * ety) / f.noise.sigma_y,
        wrap_angle(dtheta - mt) / f.noise.sigma_theta,
    )
}

fn between_residual_jacobians(
    f: &BetweenFactorPose2,
    p1: &Pose2,
    p2: &Pose2,
) -> (Vector3<f64>, Matrix3<f64>, Matrix3<f64>) {
    let (x1, y1, t1, c1, s1) = pose2_components(p1);
    let (x2, y2, t2, _, _) = pose2_components(p2);
    let (mx, my, mt, cm, sm) = pose2_components(&f.measured);

    let dx = x2 - x1;
    let dy = y2 - y1;
    let tx = c1 * dx + s1 * dy;
    let ty = -s1 * dx + c1 * dy;
    let dtheta = wrap_angle(t2 - t1);

    let etx = tx - mx;
    let ety = ty - my;
    let r0 = (cm * etx + sm * ety) / f.noise.sigma_x;
    let r1 = (-sm * etx + cm * ety) / f.noise.sigma_y;
    let r2 = wrap_angle(dtheta - mt) / f.noise.sigma_theta;

    let mut j1 = Matrix3::<f64>::zeros();
    let mut j2 = Matrix3::<f64>::zeros();

    let dr_trans = |dtx: f64, dty: f64| -> (f64, f64) {
        (
            (cm * dtx + sm * dty) / f.noise.sigma_x,
            (-sm * dtx + cm * dty) / f.noise.sigma_y,
        )
    };

    let (a, b) = dr_trans(-c1, s1);
    j1[(0, 0)] = a;
    j1[(1, 0)] = b;
    let (a, b) = dr_trans(-s1, -c1);
    j1[(0, 1)] = a;
    j1[(1, 1)] = b;
    let (a, b) = dr_trans(ty, -tx);
    j1[(0, 2)] = a;
    j1[(1, 2)] = b;
    j1[(2, 2)] = -1.0 / f.noise.sigma_theta;

    let (a, b) = dr_trans(c1, -s1);
    j2[(0, 0)] = a;
    j2[(1, 0)] = b;
    let (a, b) = dr_trans(s1, c1);
    j2[(0, 1)] = a;
    j2[(1, 1)] = b;
    j2[(2, 2)] = 1.0 / f.noise.sigma_theta;

    (Vector3::new(r0, r1, r2), j1, j2)
}

fn total_error_indexed(
    priors: &[PriorFactorPose2Indexed],
    between: &[BetweenFactorPose2Indexed],
    poses: &[Pose2],
) -> f64 {
    let mut total = 0.0;
    for f in priors {
        let r = prior_residual(
            &PriorFactorPose2 {
                key: Key(0),
                prior: f.prior,
                noise: f.noise,
            },
            &poses[f.slot],
        );
        total += 0.5 * r.norm_squared();
    }
    for f in between {
        let r = between_residual(
            &BetweenFactorPose2 {
                key1: Key(0),
                key2: Key(1),
                measured: f.measured,
                noise: f.noise,
            },
            &poses[f.slot1],
            &poses[f.slot2],
        );
        total += 0.5 * r.norm_squared();
    }
    total
}

fn linearize_system_indexed(
    n_blocks: usize,
    priors: &[PriorFactorPose2Indexed],
    between: &[BetweenFactorPose2Indexed],
    poses: &[Pose2],
) -> (SparseBlockSystem3, f64) {
    let mut system = SparseBlockSystem3::with_block_capacity(n_blocks, priors.len() + between.len() * 3);
    let mut err = 0.0;

    for f in priors {
        if let Some(i) = f.var {
            let (r, j) = prior_residual_jacobian(
                &PriorFactorPose2 {
                    key: Key(0),
                    prior: f.prior,
                    noise: f.noise,
                },
                &poses[f.slot],
            );
            err += 0.5 * r.norm_squared();
            let h_ii = j.transpose() * j;
            let b_i = j.transpose() * r;
            system.add_h(i, i, &h_ii);
            system.add_b(i, b_i);
        } else {
            err += 0.5
                * prior_residual(
                    &PriorFactorPose2 {
                        key: Key(0),
                        prior: f.prior,
                        noise: f.noise,
                    },
                    &poses[f.slot],
                )
                .norm_squared();
        }
    }

    for f in between {
        match (f.var1, f.var2) {
            (Some(i), Some(j)) => {
                let (r, j1, j2) = between_residual_jacobians(
                    &BetweenFactorPose2 {
                        key1: Key(0),
                        key2: Key(1),
                        measured: f.measured,
                        noise: f.noise,
                    },
                    &poses[f.slot1],
                    &poses[f.slot2],
                );
                err += 0.5 * r.norm_squared();
                system.add_h(i, i, &(j1.transpose() * j1));
                system.add_b(i, j1.transpose() * r);
                system.add_h(j, j, &(j2.transpose() * j2));
                system.add_b(j, j2.transpose() * r);
                system.add_h(i, j, &(j1.transpose() * j2));
            }
            (Some(i), None) => {
                let (r, j1, _) = between_residual_jacobians(
                    &BetweenFactorPose2 {
                        key1: Key(0),
                        key2: Key(1),
                        measured: f.measured,
                        noise: f.noise,
                    },
                    &poses[f.slot1],
                    &poses[f.slot2],
                );
                err += 0.5 * r.norm_squared();
                system.add_h(i, i, &(j1.transpose() * j1));
                system.add_b(i, j1.transpose() * r);
            }
            (None, Some(j)) => {
                let (r, _, j2) = between_residual_jacobians(
                    &BetweenFactorPose2 {
                        key1: Key(0),
                        key2: Key(1),
                        measured: f.measured,
                        noise: f.noise,
                    },
                    &poses[f.slot1],
                    &poses[f.slot2],
                );
                err += 0.5 * r.norm_squared();
                system.add_h(j, j, &(j2.transpose() * j2));
                system.add_b(j, j2.transpose() * r);
            }
            (None, None) => {
                err += 0.5
                    * between_residual(
                        &BetweenFactorPose2 {
                            key1: Key(0),
                            key2: Key(1),
                            measured: f.measured,
                            noise: f.noise,
                        },
                        &poses[f.slot1],
                        &poses[f.slot2],
                    )
                    .norm_squared();
            }
        }
    }

    (system, err)
}

fn total_error3(graph: &PoseGraph3, values: &BTreeMap<Key, Pose3>) -> f64 {
    let mut total = 0.0;
    for f in &graph.priors {
        if let Some(p) = values.get(&f.key) {
            total += 0.5 * f.error(p).norm_squared();
        }
    }
    for f in &graph.between {
        if let (Some(p1), Some(p2)) = (values.get(&f.key1), values.get(&f.key2)) {
            total += 0.5 * f.error(p1, p2).norm_squared();
        }
    }
    total
}

fn linearize_system3(
    graph: &PoseGraph3,
    values: &BTreeMap<Key, Pose3>,
    key_to_var: &BTreeMap<Key, usize>,
) -> (SparseBlockSystem6, f64) {
    let n_blocks = key_to_var.len();
    let mut system = SparseBlockSystem6::with_block_capacity(n_blocks, graph.priors.len() + graph.between.len() * 3);
    let mut err = 0.0;

    for f in &graph.priors {
        let estimate = values.get(&f.key).expect("value missing for prior");
        let (r, j) = prior3_residual_jacobian_numeric(f, estimate);
        err += 0.5 * r.norm_squared();
        if let Some(i) = key_to_var.get(&f.key).copied() {
            system.add_h(i, i, &(j.transpose() * j));
            system.add_b(i, j.transpose() * r);
        }
    }

    for f in &graph.between {
        let p1 = values.get(&f.key1).expect("value missing for key1");
        let p2 = values.get(&f.key2).expect("value missing for key2");
        let (r, j1, j2) = between3_residual_jacobians_numeric(f, p1, p2);
        err += 0.5 * r.norm_squared();
        let i1 = key_to_var.get(&f.key1).copied();
        let i2 = key_to_var.get(&f.key2).copied();

        match (i1, i2) {
            (Some(i), Some(j)) => {
                system.add_h(i, i, &(j1.transpose() * j1));
                system.add_b(i, j1.transpose() * r);
                system.add_h(j, j, &(j2.transpose() * j2));
                system.add_b(j, j2.transpose() * r);
                system.add_h(i, j, &(j1.transpose() * j2));
            }
            (Some(i), None) => {
                system.add_h(i, i, &(j1.transpose() * j1));
                system.add_b(i, j1.transpose() * r);
            }
            (None, Some(j)) => {
                system.add_h(j, j, &(j2.transpose() * j2));
                system.add_b(j, j2.transpose() * r);
            }
            (None, None) => {}
        }
    }

    (system, err)
}

fn variable_edges(graph: &PoseGraph2, key_to_var: &BTreeMap<Key, usize>) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for f in &graph.between {
        if let (Some(&i), Some(&j)) = (key_to_var.get(&f.key1), key_to_var.get(&f.key2)) {
            edges.push((i, j));
        }
    }
    edges
}

fn variable_edges3(graph: &PoseGraph3, key_to_var: &BTreeMap<Key, usize>) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for f in &graph.between {
        if let (Some(&i), Some(&j)) = (key_to_var.get(&f.key1), key_to_var.get(&f.key2)) {
            edges.push((i, j));
        }
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
    use gtsam_rs_math::Rot3;

    #[test]
    fn prior_factor_zero_error_on_prior() {
        let factor = PriorFactorPose3::new(Key(0), Pose3::identity(), Noise6::isotropic(1.0));
        assert!(factor.error(&Pose3::identity()).norm() < 1e-12);
    }

    #[test]
    fn between_pose2_jacobian_matches_finite_difference() {
        let f = BetweenFactorPose2::new(
            Key(0),
            Key(1),
            Pose2::new(Rot2::from_angle(0.2), Vector2::new(1.0, 0.3)),
            Noise3::isotropic(1.0),
        );
        let p1 = Pose2::new(Rot2::from_angle(0.1), Vector2::new(0.3, -0.4));
        let p2 = Pose2::new(Rot2::from_angle(0.2), Vector2::new(1.4, 0.1));
        let (_, j1, j2) = between_residual_jacobians(&f, &p1, &p2);
        let eps = 1e-6;
        for c in 0..3 {
            let mut dv = SVector::<f64, 3>::zeros();
            dv[c] = eps;
            let p1p = p1.retract(&dv);
            let p2p = p2.retract(&dv);
            let r0 = between_residual(&f, &p1, &p2);
            let r1 = between_residual(&f, &p1p, &p2);
            let r2 = between_residual(&f, &p1, &p2p);
            assert!(((r1 - r0) / eps - j1.column(c)).norm() < 1e-4);
            assert!(((r2 - r0) / eps - j2.column(c)).norm() < 1e-4);
        }
    }

    #[test]
    fn min_degree_ordering_is_permutation() {
        let mut graph = PoseGraph2::new();
        let noise = Noise3::isotropic(0.1);
        graph.add_prior(PriorFactorPose2::new(
            Key(0),
            Pose2::new(Rot2::from_angle(0.0), Vector2::zeros()),
            noise,
        ));
        graph.add_between(BetweenFactorPose2::new(
            Key(0),
            Key(1),
            Pose2::new(Rot2::from_angle(0.0), Vector2::new(1.0, 0.0)),
            noise,
        ));
        graph.add_between(BetweenFactorPose2::new(
            Key(1),
            Key(2),
            Pose2::new(Rot2::from_angle(0.0), Vector2::new(1.0, 0.0)),
            noise,
        ));
        let key_to_var = BTreeMap::from([(Key(1), 0usize), (Key(2), 1usize)]);
        let mut ord = compute_ordering_from_edges(2, &variable_edges(&graph, &key_to_var), OrderingStrategy::MinDegree);
        ord.sort_unstable();
        assert_eq!(ord, vec![0, 1]);
    }

    #[test]
    fn sparse_elimination_matches_dense_solution() {
        let mut sys = SparseBlockSystem3::new(3);
        let eye = Matrix3::<f64>::identity();
        sys.add_h(0, 0, &(2.0 * eye));
        sys.add_h(1, 1, &(3.0 * eye));
        sys.add_h(2, 2, &(4.0 * eye));
        sys.add_h(0, 1, &(0.2 * eye));
        sys.add_h(1, 2, &(-0.1 * eye));
        sys.add_b(0, Vector3::new(1.0, -2.0, 0.5));
        sys.add_b(1, Vector3::new(-0.2, 0.3, 0.1));
        sys.add_b(2, Vector3::new(0.7, 0.2, -0.8));

        let ord = compute_ordering_from_edges(3, &[(0, 1), (1, 2)], OrderingStrategy::MinDegree);
        let xs = sys.solve(&ord, 1e-9).expect("sparse solve");

        let mut h = DMatrix::<f64>::zeros(9, 9);
        let mut b = DVector::<f64>::zeros(9);
        for i in 0..3 {
            b.rows_mut(i * 3, 3).copy_from(&sys.b_block(i));
            for j in i..3 {
                let blk = sys.get_h(i, j);
                for r in 0..3 {
                    for c in 0..3 {
                        h[(i * 3 + r, j * 3 + c)] = blk[(r, c)];
                        h[(j * 3 + c, i * 3 + r)] = blk[(r, c)];
                    }
                }
            }
        }
        for i in 0..9 {
            h[(i, i)] += 1e-9;
        }
        let xd = h.lu().solve(&(-b)).expect("dense solve");
        assert!((xs - xd).norm() < 1e-7);
    }

    #[test]
    fn sparse6_elimination_matches_dense_pose3_linearization() {
        let mut sys = SparseBlockSystem6::new(2);
        let eye = SMatrix::<f64, 6, 6>::identity();
        sys.add_h(0, 0, &(2.0 * eye));
        sys.add_h(1, 1, &(3.0 * eye));
        sys.add_h(0, 1, &(0.1 * eye));
        sys.add_b(0, Vec6::new(1.0, -2.0, 0.5, 0.1, -0.1, 0.2));
        sys.add_b(1, Vec6::new(-0.2, 0.3, 0.1, -0.3, 0.4, -0.5));

        let ord = compute_ordering_from_edges(2, &[(0, 1)], OrderingStrategy::MinDegree);
        let xs = sys.solve(&ord, 1e-9).expect("sparse solve");

        let mut h = DMatrix::<f64>::zeros(12, 12);
        let mut b = DVector::<f64>::zeros(12);
        for i in 0..2 {
            b.rows_mut(i * 6, 6).copy_from(&sys.b_block(i));
            for j in i..2 {
                let blk = sys.get_h(i, j);
                for r in 0..6 {
                    for c in 0..6 {
                        h[(i * 6 + r, j * 6 + c)] = blk[(r, c)];
                        h[(j * 6 + c, i * 6 + r)] = blk[(r, c)];
                    }
                }
            }
        }
        for i in 0..12 {
            h[(i, i)] += 1e-9;
        }
        let xd = h.lu().solve(&(-b)).expect("dense solve");
        assert!((xs - xd).norm() < 1e-7);
    }

    #[test]
    fn pose_graph2_solves_chain() {
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
        let expected = Pose2::new(Rot2::from_angle(0.0), Vector2::new(2.0, 0.0));
        let err = expected.inverse().compose(x2_est);
        assert!(err.translation.norm() < 1e-2);
        assert!(err.rotation.angle().abs() < 1e-2);
    }

    #[test]
    fn pose_graph3_solves_chain() {
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
}

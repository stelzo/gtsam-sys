#![forbid(unsafe_code)]

use gtsam_rs_inference::{FactorGraph, Key, Values};
use gtsam_rs_linear::GaussianFactorGraph;

pub trait NonlinearFactor: Send + Sync {
    fn keys(&self) -> &[Key];

    fn linearize(&self, _values: &Values<Vec<f64>>) -> Option<GaussianFactorGraph> {
        None
    }

    fn error(&self, _values: &Values<Vec<f64>>) -> f64 {
        0.0
    }
}

#[derive(Default)]
pub struct NonlinearFactorGraph {
    factors: FactorGraph<Box<dyn NonlinearFactor>>,
}

impl NonlinearFactorGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_factor<F>(&mut self, factor: F)
    where
        F: NonlinearFactor + 'static,
    {
        self.factors.add_factor(Box::new(factor));
    }

    pub fn len(&self) -> usize {
        self.factors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct OptimizerSummary {
    pub iterations: usize,
    pub final_error: f64,
    pub converged: bool,
}

pub trait Optimizer {
    type Output;

    fn optimize(&self, graph: &NonlinearFactorGraph, initial: &Values<Vec<f64>>) -> Self::Output;
}

#[derive(Clone, Debug)]
pub struct GaussNewton {
    pub max_iterations: usize,
}

impl Default for GaussNewton {
    fn default() -> Self {
        Self { max_iterations: 50 }
    }
}

impl Optimizer for GaussNewton {
    type Output = Result<OptimizerSummary, String>;

    fn optimize(&self, graph: &NonlinearFactorGraph, initial: &Values<Vec<f64>>) -> Self::Output {
        let _ = initial;
        Ok(OptimizerSummary {
            iterations: 0,
            final_error: graph
                .factors
                .iter()
                .map(|f| f.error(initial))
                .sum::<f64>(),
            converged: true,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct LevenbergMarquardt {
    pub max_iterations: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Dogleg {
    pub max_iterations: usize,
}

pub type GaussNewtonOptimizer = GaussNewton;
pub type LevenbergMarquardtOptimizer = LevenbergMarquardt;
pub type DoglegOptimizer = Dogleg;

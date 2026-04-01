#![forbid(unsafe_code)]

pub mod nonlinear {
    pub use gtsam_rs_nonlinear::{DoglegOptimizer, GaussNewtonOptimizer, LevenbergMarquardtOptimizer};
}

pub mod graph {
    pub use gtsam_rs_inference::{Key, Symbol, Values};
}

pub mod navigation {
    pub use gtsam_rs_navigation::{ImuIntegration, ImuParameters};
}

pub mod slam {
    pub use gtsam_rs_slam::{
        BetweenFactor, BetweenFactorPose2, BetweenFactorPose3, GaussNewtonPoseGraph2,
        GaussNewtonPoseGraph3, PoseGraph2, PoseGraph3, PriorFactor, PriorFactorPose2,
        PriorFactorPose3,
    };
}

pub use gtsam_rs_inference as inference;
pub use gtsam_rs_navigation as nav;
pub use gtsam_rs_nonlinear as nonlinear_core;
pub use gtsam_rs_slam as sam;

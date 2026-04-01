#![forbid(unsafe_code)]

pub mod geometry {
    pub use gtsam_rs_math::{
        LieGroup, LocalCoordinates, Manifold, Point2, Point3, Pose2, Pose3, Retract, Rot2, Rot3,
        Unit3,
    };
}

pub mod noise {
    #[derive(Clone, Debug)]
    pub struct Diagonal;

    #[derive(Clone, Debug)]
    pub struct Isotropic;

    #[derive(Clone, Debug)]
    pub struct Gaussian;

    #[derive(Clone, Debug)]
    pub struct Robust;
}

pub mod graph {
    pub use gtsam_rs_inference::{Factor, FactorGraph, Key, Symbol, Values};
    pub use gtsam_rs_linear::{
        BayesTreeClique, GaussianBayesTree, GaussianConditional, Ordering, OrderingStrategy,
        SparseBlockSystem, SparseBlockSystem3, SparseBlockSystem6,
    };
    pub use gtsam_rs_nonlinear::{NonlinearFactor, NonlinearFactorGraph};
}

pub mod optimize {
    pub use gtsam_rs_nonlinear::{
        Dogleg, GaussNewton, LevenbergMarquardt, Optimizer, OptimizerSummary,
    };
}

pub mod navigation {
    pub use gtsam_rs_navigation::{
        CombinedImuFactor, ImuBias, ImuIntegration, ImuParameters, IntegrationState, NavState,
        PreintegratedImuMeasurements,
    };
}

pub mod slam {
    pub use gtsam_rs_slam::{
        BearingRangeFactor, BetweenFactor, BetweenFactorPose2, GaussNewtonPoseGraph2,
        BetweenFactorPose3, GaussNewtonPoseGraph3, GenericProjectionFactor, Noise3, Noise6,
        PoseGraph2, PoseGraph2Result, PoseGraph2Summary, PoseGraph3, PoseGraph3Result,
        PoseGraph3Summary, PriorFactor, PriorFactorPose2, PriorFactorPose3,
    };
}

pub mod compat {
    pub use gtsam_rs_compat::*;
}

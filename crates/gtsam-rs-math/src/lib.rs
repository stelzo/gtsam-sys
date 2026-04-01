#![forbid(unsafe_code)]

use nalgebra::{
    Matrix3, Point2 as NaPoint2, Point3 as NaPoint3, Rotation2, SVector, Unit, UnitQuaternion,
    Vector2, Vector3,
};

pub trait Retract: Sized {
    type Tangent;

    fn retract(&self, tangent: &Self::Tangent) -> Self;
}

pub trait LocalCoordinates {
    type Tangent;

    fn local_coordinates(&self, other: &Self) -> Self::Tangent;
}

pub trait Manifold: Retract + LocalCoordinates {}

impl<T> Manifold for T where T: Retract + LocalCoordinates {}

pub trait LieGroup: Manifold {
    fn identity() -> Self;
    fn compose(&self, other: &Self) -> Self;
    fn inverse(&self) -> Self;
}

pub type Point2 = NaPoint2<f64>;
pub type Point3 = NaPoint3<f64>;

impl Retract for Point2 {
    type Tangent = Vector2<f64>;

    fn retract(&self, tangent: &Self::Tangent) -> Self {
        *self + *tangent
    }
}

impl LocalCoordinates for Point2 {
    type Tangent = Vector2<f64>;

    fn local_coordinates(&self, other: &Self) -> Self::Tangent {
        other - self
    }
}

impl Retract for Point3 {
    type Tangent = Vector3<f64>;

    fn retract(&self, tangent: &Self::Tangent) -> Self {
        *self + *tangent
    }
}

impl LocalCoordinates for Point3 {
    type Tangent = Vector3<f64>;

    fn local_coordinates(&self, other: &Self) -> Self::Tangent {
        other - self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Unit3(pub Unit<Vector3<f64>>);

impl Unit3 {
    pub fn new_normalize(v: Vector3<f64>) -> Self {
        Self(Unit::new_normalize(v))
    }

    pub fn as_ref(&self) -> &Unit<Vector3<f64>> {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rot2(pub Rotation2<f64>);

impl Rot2 {
    pub fn from_angle(theta: f64) -> Self {
        Self(Rotation2::new(theta))
    }

    pub fn angle(&self) -> f64 {
        self.0.angle()
    }
}

impl Retract for Rot2 {
    type Tangent = f64;

    fn retract(&self, tangent: &Self::Tangent) -> Self {
        Self(self.0 * Rotation2::new(*tangent))
    }
}

impl LocalCoordinates for Rot2 {
    type Tangent = f64;

    fn local_coordinates(&self, other: &Self) -> Self::Tangent {
        (self.0.inverse() * other.0).angle()
    }
}

impl LieGroup for Rot2 {
    fn identity() -> Self {
        Self(Rotation2::identity())
    }

    fn compose(&self, other: &Self) -> Self {
        Self(self.0 * other.0)
    }

    fn inverse(&self) -> Self {
        Self(self.0.inverse())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rot3(pub UnitQuaternion<f64>);

impl Rot3 {
    pub fn from_quaternion(q: UnitQuaternion<f64>) -> Self {
        Self(q)
    }

    pub fn from_scaled_axis(axis_angle: Vector3<f64>) -> Self {
        Self(UnitQuaternion::from_scaled_axis(axis_angle))
    }

    pub fn scaled_axis(&self) -> Vector3<f64> {
        self.0.scaled_axis()
    }

    pub fn as_quaternion(&self) -> &UnitQuaternion<f64> {
        &self.0
    }
}

impl Retract for Rot3 {
    type Tangent = Vector3<f64>;

    fn retract(&self, tangent: &Self::Tangent) -> Self {
        Self(self.0 * UnitQuaternion::from_scaled_axis(*tangent))
    }
}

impl LocalCoordinates for Rot3 {
    type Tangent = Vector3<f64>;

    fn local_coordinates(&self, other: &Self) -> Self::Tangent {
        (self.0.inverse() * other.0).scaled_axis()
    }
}

impl LieGroup for Rot3 {
    fn identity() -> Self {
        Self(UnitQuaternion::identity())
    }

    fn compose(&self, other: &Self) -> Self {
        Self(self.0 * other.0)
    }

    fn inverse(&self) -> Self {
        Self(self.0.inverse())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose2 {
    pub rotation: Rot2,
    pub translation: Vector2<f64>,
}

impl Pose2 {
    pub fn new(rotation: Rot2, translation: Vector2<f64>) -> Self {
        Self {
            rotation,
            translation,
        }
    }
}

impl Retract for Pose2 {
    type Tangent = SVector<f64, 3>;

    fn retract(&self, tangent: &Self::Tangent) -> Self {
        Self {
            rotation: self.rotation.retract(&tangent[2]),
            translation: self.translation + Vector2::new(tangent[0], tangent[1]),
        }
    }
}

impl LocalCoordinates for Pose2 {
    type Tangent = SVector<f64, 3>;

    fn local_coordinates(&self, other: &Self) -> Self::Tangent {
        SVector::<f64, 3>::new(
            other.translation.x - self.translation.x,
            other.translation.y - self.translation.y,
            self.rotation.local_coordinates(&other.rotation),
        )
    }
}

impl LieGroup for Pose2 {
    fn identity() -> Self {
        Self {
            rotation: Rot2::identity(),
            translation: Vector2::zeros(),
        }
    }

    fn compose(&self, other: &Self) -> Self {
        Self {
            rotation: self.rotation.compose(&other.rotation),
            translation: self.translation + self.rotation.0 * other.translation,
        }
    }

    fn inverse(&self) -> Self {
        let inv_r = self.rotation.inverse();
        Self {
            rotation: inv_r,
            translation: -(inv_r.0 * self.translation),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose3 {
    pub rotation: Rot3,
    pub translation: Vector3<f64>,
}

impl Pose3 {
    pub fn new(rotation: Rot3, translation: Vector3<f64>) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    fn skew(v: Vector3<f64>) -> Matrix3<f64> {
        Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
    }

    fn so3_left_jacobian(phi: Vector3<f64>) -> Matrix3<f64> {
        let theta = phi.norm();
        let omega = Self::skew(phi);
        let omega2 = omega * omega;
        let i = Matrix3::<f64>::identity();
        if theta < 1e-9 {
            i + 0.5 * omega + (1.0 / 6.0) * omega2
        } else {
            let theta2 = theta * theta;
            let a = (1.0 - theta.cos()) / theta2;
            let b = (theta - theta.sin()) / (theta2 * theta);
            i + a * omega + b * omega2
        }
    }

    fn so3_left_jacobian_inverse(phi: Vector3<f64>) -> Matrix3<f64> {
        let theta = phi.norm();
        let omega = Self::skew(phi);
        let omega2 = omega * omega;
        let i = Matrix3::<f64>::identity();
        if theta < 1e-9 {
            i - 0.5 * omega + (1.0 / 12.0) * omega2
        } else {
            let sin_theta = theta.sin();
            if sin_theta.abs() < 1e-12 {
                i - 0.5 * omega + (1.0 / 12.0) * omega2
            } else {
                let theta2 = theta * theta;
                let c = (1.0 / theta2) - ((1.0 + theta.cos()) / (2.0 * theta * sin_theta));
                i - 0.5 * omega + c * omega2
            }
        }
    }
}

impl Retract for Pose3 {
    type Tangent = SVector<f64, 6>;

    fn retract(&self, tangent: &Self::Tangent) -> Self {
        let rho = Vector3::new(tangent[0], tangent[1], tangent[2]);
        let phi = Vector3::new(tangent[3], tangent[4], tangent[5]);
        let delta_r = Rot3::from_scaled_axis(phi);
        let delta_t = Self::so3_left_jacobian(phi) * rho;
        self.compose(&Self {
            rotation: delta_r,
            translation: delta_t,
        })
    }
}

impl LocalCoordinates for Pose3 {
    type Tangent = SVector<f64, 6>;

    fn local_coordinates(&self, other: &Self) -> Self::Tangent {
        let delta = self.inverse().compose(other);
        let phi = delta.rotation.scaled_axis();
        let rho = Self::so3_left_jacobian_inverse(phi) * delta.translation;
        SVector::<f64, 6>::new(rho.x, rho.y, rho.z, phi.x, phi.y, phi.z)
    }
}

impl LieGroup for Pose3 {
    fn identity() -> Self {
        Self {
            rotation: Rot3::identity(),
            translation: Vector3::zeros(),
        }
    }

    fn compose(&self, other: &Self) -> Self {
        Self {
            rotation: self.rotation.compose(&other.rotation),
            translation: self.translation + self.rotation.0 * other.translation,
        }
    }

    fn inverse(&self) -> Self {
        let inv_r = self.rotation.inverse();
        Self {
            rotation: inv_r,
            translation: -(inv_r.0 * self.translation),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{LieGroup, LocalCoordinates, Point2, Pose3, Retract, Rot3};
    use nalgebra::{SVector, Vector2, Vector3};

    const EPS: f64 = 1e-9;

    fn assert_vec3_close(a: Vector3<f64>, b: Vector3<f64>) {
        assert!((a - b).norm() < EPS, "left={a:?} right={b:?}");
    }

    fn assert_rot3_close(a: Rot3, b: Rot3) {
        let delta = a.local_coordinates(&b);
        assert!(delta.norm() < EPS, "delta={delta:?}");
    }

    #[test]
    fn rot3_group_identity_inverse() {
        let r = Rot3::from_scaled_axis(Vector3::new(0.2, -0.4, 0.1));
        assert_rot3_close(Rot3::identity().compose(&r), r);
        assert_rot3_close(r.compose(&r.inverse()), Rot3::identity());
    }

    #[test]
    fn rot3_retract_local_roundtrip() {
        let r0 = Rot3::from_scaled_axis(Vector3::new(-0.1, 0.3, 0.2));
        let delta = Vector3::new(0.01, -0.02, 0.03);
        let r1 = r0.retract(&delta);
        let recovered = r0.local_coordinates(&r1);
        assert_vec3_close(delta, recovered);
    }

    #[test]
    fn pose3_group_identity_inverse() {
        let p = Pose3::new(
            Rot3::from_scaled_axis(Vector3::new(0.1, -0.2, 0.3)),
            Vector3::new(1.0, -2.0, 3.0),
        );

        let id = Pose3::identity();
        let composed = id.compose(&p);
        assert_rot3_close(composed.rotation, p.rotation);
        assert_vec3_close(composed.translation, p.translation);

        let back = p.compose(&p.inverse());
        assert_rot3_close(back.rotation, Pose3::identity().rotation);
        assert_vec3_close(back.translation, Vector3::zeros());
    }

    #[test]
    fn pose3_retract_local_roundtrip() {
        let p0 = Pose3::new(
            Rot3::from_scaled_axis(Vector3::new(0.2, 0.1, -0.1)),
            Vector3::new(4.0, -1.0, 0.5),
        );

        let delta = SVector::<f64, 6>::new(0.5, -0.2, 0.1, 0.03, -0.01, 0.02);
        let p1 = p0.retract(&delta);
        let recovered = p0.local_coordinates(&p1);
        assert!((recovered - delta).norm() < EPS, "recovered={recovered:?}");
    }

    #[test]
    fn rot3_near_pi_is_finite() {
        let near_pi = Vector3::new(std::f64::consts::PI - 1e-8, 0.0, 0.0);
        let r = Rot3::from_scaled_axis(near_pi);
        let back = r.scaled_axis();
        assert!(back.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn manifold_default_chart_point2() {
        // Port of the Point2 local/retract assertions in gtsam/tests/testManifold.cpp.
        let p0 = Point2::new(0.0, 0.0);
        let p1 = Point2::new(1.0, 0.0);
        let local = p0.local_coordinates(&p1);
        assert_eq!(local, Vector2::new(1.0, 0.0));
        let retracted = p0.retract(&local);
        assert_eq!(retracted, p1);
    }

    #[test]
    fn manifold_default_chart_rot3_identity() {
        // Port of the Rot3 chart test shape from gtsam/tests/testManifold.cpp.
        let id = Rot3::identity();
        let v = Vector3::new(1.0, 1.0, 1.0);
        let r = id.retract(&v);
        let back = id.local_coordinates(&r);
        assert_vec3_close(back, v);
        assert_vec3_close(r.local_coordinates(&r), Vector3::zeros());
    }
}

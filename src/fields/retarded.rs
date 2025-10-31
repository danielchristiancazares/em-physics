use crate::constants::{SPEED_OF_LIGHT, VACUUM_PERMEABILITY};
use crate::math::{R3, Scalar};

use super::sources::WireSegment3D;

type TimeFn = Box<dyn Fn(Scalar) -> Scalar + Send + Sync + 'static>;

/// Thin current element (wire segment) carrying a real time-varying current i(t).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TimeLineCurrent {
    /// Wire geometry.
    pub segment: WireSegment3D,
    /// Current function i(t) in amperes.
    pub current: TimeFn,
    /// Optional derivative di/dt if available for E-field evaluation.
    pub current_derivative: Option<TimeFn>,
}

impl TimeLineCurrent {
    /// Creates a line current with optional derivative function.
    #[must_use]
    pub fn new<F>(segment: WireSegment3D, current: F) -> Self
    where
        F: Fn(Scalar) -> Scalar + Send + Sync + 'static,
    {
        Self { segment, current: Box::new(current), current_derivative: None }
    }

    /// Attaches an analytic derivative function.
    pub fn with_derivative<F>(mut self, deriv: F) -> Self
    where
        F: Fn(Scalar) -> Scalar + Send + Sync + 'static,
    {
        self.current_derivative = Some(Box::new(deriv));
        self
    }
}

/// Computes the retarded-time vector potential A(r, t) from thin wire segments using midpoint quadrature.
#[must_use]
pub fn vector_potential_retarded(point: R3, time: Scalar, lines: &[TimeLineCurrent], samples_per_segment: usize) -> R3 {
    let mu = VACUUM_PERMEABILITY;
    let coeff = mu / (4.0 * std::f64::consts::PI);
    let mut a = R3::zeros();
    for lc in lines {
        let m = samples_per_segment.max(1);
        let dl = (lc.segment.end - lc.segment.start) / m as Scalar;
        for k in 0..m {
            let mid = lc.segment.start + dl * (k as Scalar + 0.5);
            let r_vec = point - mid;
            let r = r_vec.norm();
            if r <= 1.0e-12 { continue; }
            let t_ret = time - r / SPEED_OF_LIGHT;
            let i = (lc.current)(t_ret);
            a += dl * (coeff * i / r);
        }
    }
    a
}

/// Approximates the electric field via E ≈ -∂A/∂t using either an analytic derivative of the current
/// or central differences on A with a small `dt`.
#[must_use]
pub fn electric_field_from_retarded_potential(point: R3, time: Scalar, lines: &[TimeLineCurrent], samples_per_segment: usize, dt: Option<Scalar>) -> R3 {
    // Prefer analytic derivative if provided for all lines.
    let all_have_deriv = lines.iter().all(|lc| lc.current_derivative.is_some());
    if all_have_deriv {
        let mu = VACUUM_PERMEABILITY;
        let coeff = mu / (4.0 * std::f64::consts::PI);
        let mut e = R3::zeros();
        for lc in lines {
            let m = samples_per_segment.max(1);
            let dl = (lc.segment.end - lc.segment.start) / m as Scalar;
            for k in 0..m {
                let mid = lc.segment.start + dl * (k as Scalar + 0.5);
                let r_vec = point - mid;
                let r = r_vec.norm();
                if r <= 1.0e-12 { continue; }
                let t_ret = time - r / SPEED_OF_LIGHT;
                let di_dt = (lc.current_derivative.as_ref().unwrap())(t_ret);
                // E ≈ -∂A/∂t = -(μ0/4π) (di/dt) ∫ dl / r
                e += dl * (-coeff * di_dt / r);
            }
        }
        return e;
    }
    // Fallback: finite-difference A.
    let dt = dt.unwrap_or(1.0e-12);
    let a_plus = vector_potential_retarded(point, time + dt, lines, samples_per_segment);
    let a_minus = vector_potential_retarded(point, time - dt, lines, samples_per_segment);
    (a_plus - a_minus) * (-0.5 / dt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_potential_zero_far_away_with_zero_current() {
        let seg = WireSegment3D { start: R3::new(0.0, 0.0, 0.0), end: R3::new(1.0, 0.0, 0.0) };
        let lc = TimeLineCurrent::new(seg, |_t| 0.0);
        let a = vector_potential_retarded(R3::new(10.0, 0.0, 0.0), 0.0, &[lc], 10);
        assert!(a.norm() <= 1.0e-15);
    }
}



use crate::constants::VACUUM_PERMITTIVITY;
use crate::math::{R3, Scalar};

/// Point charge in coulombs.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointCharge {
    /// Position in meters.
    pub position: R3,
    /// Charge in coulombs.
    pub charge_c: Scalar,
}

#[inline]
fn coulomb_coeff() -> Scalar {
    1.0 / (4.0 * std::f64::consts::PI * VACUUM_PERMITTIVITY)
}

/// Electric potential φ at `point` due to discrete point charges.
#[must_use]
pub fn potential_from_point_charges(point: R3, charges: &[PointCharge]) -> Scalar {
    let k = coulomb_coeff();
    let mut phi = 0.0;
    for c in charges {
        let r = (point - c.position).norm();
        if r > 1.0e-12 {
            phi += k * c.charge_c / r;
        }
    }
    phi
}

/// Electric field E at `point` due to discrete point charges.
#[must_use]
pub fn electric_field_from_point_charges(point: R3, charges: &[PointCharge]) -> R3 {
    let k = coulomb_coeff();
    let mut e = R3::zeros();
    for c in charges {
        let r_vec = point - c.position;
        let r = r_vec.norm();
        if r > 1.0e-12 {
            e += r_vec * (k * c.charge_c / (r * r * r));
        }
    }
    e
}

/// Approximates potential of a uniform surface charge density `sigma_c_per_m2` on a flat patch
/// by collapsing the patch to a point charge located at its centroid with `q = σ A`.
#[must_use]
pub fn potential_from_uniform_patch(point: R3, centroid: R3, area: Scalar, sigma_c_per_m2: Scalar) -> Scalar {
    let q = sigma_c_per_m2 * area;
    potential_from_point_charges(point, &[PointCharge { position: centroid, charge_c: q }])
}

/// Approximates electric field of a uniform surface charge density on a flat patch using a
/// point-charge collapse at the centroid.
#[must_use]
pub fn electric_field_from_uniform_patch(point: R3, centroid: R3, area: Scalar, sigma_c_per_m2: Scalar) -> R3 {
    let q = sigma_c_per_m2 * area;
    electric_field_from_point_charges(point, &[PointCharge { position: centroid, charge_c: q }])
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn potential_of_single_point_charge_matches_reference_axis() {
        let q = PointCharge { position: R3::new(0.0, 0.0, 0.0), charge_c: 1.0e-9 };
        let p = R3::new(0.0, 0.0, 1.0);
        let phi = potential_from_point_charges(p, &[q]);
        let ref_val = 1.0 / (4.0 * std::f64::consts::PI * VACUUM_PERMITTIVITY) * 1.0e-9;
        assert_relative_eq!(phi, ref_val, max_relative = 1.0e-12);
    }
}



use nalgebra::{DMatrix, DVector};

use crate::math::{R3, Scalar};

/// Flat boundary element with centroid, outward unit normal, and area.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlatPanel {
    /// Panel centroid in meters.
    pub centroid: R3,
    /// Outward unit normal.
    pub normal: R3,
    /// Panel area in square meters.
    pub area: Scalar,
}

impl FlatPanel {
    /// Creates a panel ensuring the normal is normalized.
    #[must_use]
    pub fn new(centroid: R3, normal: R3, area: Scalar) -> Self {
        let n = if normal.norm() > 0.0 { normal.normalize() } else { R3::new(0.0, 0.0, 0.0) };
        Self { centroid, normal: n, area }
    }
}

#[inline]
fn green_laplace(obs: R3, src: R3) -> Scalar {
    let r = (obs - src).norm();
    if r <= 1.0e-12 { 0.0 } else { 1.0 / (4.0 * std::f64::consts::PI * r) }
}

#[inline]
fn dgreen_dn_src(obs: R3, src: R3, n_src: R3) -> Scalar {
    let r_vec = obs - src;
    let r = r_vec.norm();
    if r <= 1.0e-12 { 0.0 } else { -(r_vec.dot(&n_src)) / (4.0 * std::f64::consts::PI * r * r * r) }
}

/// Assembles the single-layer potential matrix S for Laplace's equation using centroid quadrature.
/// S_{ij} ≈ ∫_{panel j} G(r_i, r') dS ≈ area_j * G(r_i, centroid_j).
#[must_use]
pub fn build_single_layer_matrix(panels: &[FlatPanel]) -> DMatrix<Scalar> {
    let n = panels.len();
    let mut s = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                // Diagonal regularization using an effective distance ~ sqrt(area/pi)
                let r_eff = (panels[j].area / std::f64::consts::PI).sqrt().max(1.0e-12);
                s[(i, j)] = panels[j].area / (4.0 * std::f64::consts::PI * r_eff);
            } else {
                s[(i, j)] = panels[j].area * green_laplace(panels[i].centroid, panels[j].centroid);
            }
        }
    }
    s
}

/// Assembles the double-layer operator D using centroid quadrature of ∂G/∂n'.
/// D_{ij} ≈ ∫_{panel j} ∂G/∂n'(r_i, r') dS ≈ area_j * ∂G/∂n'(centroid_j).
#[must_use]
pub fn build_double_layer_matrix(panels: &[FlatPanel]) -> DMatrix<Scalar> {
    let n = panels.len();
    let mut d = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                // Principal value on smooth surfaces tends to ±1/2; we use 0.5 as a simple model.
                d[(i, j)] = 0.5;
            } else {
                d[(i, j)] = panels[j].area * dgreen_dn_src(panels[i].centroid, panels[j].centroid, panels[j].normal);
            }
        }
    }
    d
}

/// Evaluates the single-layer potential Φ at arbitrary points from surface charge density σ on panels.
#[must_use]
pub fn potential_from_single_layer(points: &[R3], sigma: &DVector<Scalar>, panels: &[FlatPanel]) -> Vec<Scalar> {
    let mut out = Vec::with_capacity(points.len());
    for p in points {
        let mut phi = 0.0;
        for (j, panel) in panels.iter().enumerate() {
            phi += panel.area * green_laplace(*p, panel.centroid) * sigma[j];
        }
        out.push(phi);
    }
    out
}

/// Evaluates the electric field E = -∇Φ from a single-layer represented by point panels using
/// the gradient of G. This centroid approximation is valid far from the surface.
#[must_use]
pub fn electric_field_from_single_layer(points: &[R3], sigma: &DVector<Scalar>, panels: &[FlatPanel]) -> Vec<R3> {
    let mut out = Vec::with_capacity(points.len());
    for p in points {
        let mut e = R3::zeros();
        for (j, panel) in panels.iter().enumerate() {
            let r_vec = *p - panel.centroid;
            let r = r_vec.norm().max(1.0e-12);
            let grad_g = -r_vec / (4.0 * std::f64::consts::PI * r * r * r);
            e += grad_g * (panel.area * sigma[j]);
        }
        out.push(e * -1.0);
    }
    out
}

/// Convenience solver for a pure single-layer Dirichlet problem S σ = Φ (centroid collocation).
/// Returns the surface charge density σ.
#[must_use]
pub fn solve_dirichlet_single_layer(panels: &[FlatPanel], boundary_potential: &DVector<Scalar>) -> Option<DVector<Scalar>> {
    let s = build_single_layer_matrix(panels);
    s.lu().solve(boundary_potential)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn single_layer_self_regularization_is_finite() {
        let panels = [FlatPanel::new(R3::new(0.0, 0.0, 0.0), R3::new(0.0, 0.0, 1.0), 1.0)];
        let s = build_single_layer_matrix(&panels);
        assert!(s[(0,0)].is_finite());
        assert!(s[(0,0)] > 0.0);
    }

    #[test]
    fn potential_far_field_scales_as_total_charge_over_r() {
        let panels = [FlatPanel::new(R3::new(0.0, 0.0, 0.0), R3::new(0.0, 0.0, 1.0), 1.0)];
        let sigma = DVector::from_vec(vec![1.0]); // total charge ≈ 1 C (up to 1/(4π) factor)
        let p = R3::new(0.0, 0.0, 100.0);
        let phi = potential_from_single_layer(&[p], &sigma, &panels)[0];
        assert_relative_eq!(phi * 4.0 * std::f64::consts::PI * 100.0, 1.0, max_relative = 5e-2);
    }
}



//! Transmission line primitives and ABCD parameterization.

use crate::math::Scalar;

use super::twoport::{C, TwoPort};

/// Distributed RLGC parameters per unit length.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RLGC {
    /// Series resistance per meter (Ω/m).
    pub r_per_m: Scalar,
    /// Series inductance per meter (H/m).
    pub l_per_m: Scalar,
    /// Shunt conductance per meter (S/m).
    pub g_per_m: Scalar,
    /// Shunt capacitance per meter (F/m).
    pub c_per_m: Scalar,
}

impl RLGC {
    /// Lossless line parameters (R=G=0).
    #[must_use]
    pub fn lossless(l_per_m: Scalar, c_per_m: Scalar) -> Self {
        Self {
            r_per_m: 0.0,
            l_per_m,
            g_per_m: 0.0,
            c_per_m,
        }
    }
}

/// Transmission line descriptor (uniform, per-unit-length parameters).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransmissionLine {
    /// Physical length in meters.
    pub length_m: Scalar,
    /// Distributed parameters.
    pub rlgc: RLGC,
}

impl TransmissionLine {
    /// Line of length `length_m` with the given RLGC per-unit parameters.
    #[must_use]
    pub fn new(length_m: Scalar, rlgc: RLGC) -> Self {
        Self { length_m, rlgc }
    }

    /// Lossless line shortcut.
    #[must_use]
    pub fn lossless(length_m: Scalar, l_per_m: Scalar, c_per_m: Scalar) -> Self {
        Self::new(length_m, RLGC::lossless(l_per_m, c_per_m))
    }

    /// Returns ABCD two-port for angular frequency `omega`.
    #[must_use]
    pub fn to_twoport(&self, omega: Scalar) -> TwoPort {
        // γ = sqrt((R + jωL)(G + jωC)); Zc = sqrt((R + jωL)/(G + jωC))
        let j = C::new(0.0, 1.0);
        let r = self.rlgc.r_per_m;
        let l = self.rlgc.l_per_m;
        let g = self.rlgc.g_per_m;
        let c = self.rlgc.c_per_m;

        let jw = j * omega;
        let series = C::new(r, 0.0) + jw * l;
        let shunt = C::new(g, 0.0) + jw * c;

        // Guard against degenerate shunt
        if shunt.norm() == 0.0 {
            // Acts as pure series; approximate extremely large Zc
            return TwoPort::from_abcd(C::new(1.0, 0.0), C::new(series.re * self.length_m, series.im * self.length_m), C::new(0.0, 0.0), C::new(1.0, 0.0));
        }

        let gamma = (series * shunt).sqrt();
        let zc = (series / shunt).sqrt();
        let gl = gamma * self.length_m;

        // ABCD = [[cosh(γl), Zc*sinh(γl)],[sinh(γl)/Zc, cosh(γl)]]
        let a = gl.cosh();
        let s = gl.sinh();
        let b = zc * s;
        let c_elem = s / zc;
        TwoPort::from_abcd(a, b, c_elem, a)
    }
}

impl TransmissionLine {
    /// Input impedance with load `z_load` at angular frequency `omega`.
    #[must_use]
    pub fn input_impedance(&self, omega: Scalar, z_load: C) -> C {
        let j = C::new(0.0, 1.0);
        let r = self.rlgc.r_per_m;
        let l = self.rlgc.l_per_m;
        let g = self.rlgc.g_per_m;
        let c = self.rlgc.c_per_m;
        let jw = j * omega;
        let series = C::new(r, 0.0) + jw * l;
        let shunt = C::new(g, 0.0) + jw * c;
        if shunt.norm() == 0.0 {
            return z_load + C::new(series.re * self.length_m, series.im * self.length_m);
        }
        let gamma = (series * shunt).sqrt();
        let zc = (series / shunt).sqrt();
        let gl = gamma * self.length_m;
        let t = gl.tanh();
        (zc * (z_load + zc * t)) / (zc + z_load * t)
    }

    /// Reflection coefficient at port 1 for real reference impedance `z0` and load `z_load`.
    #[must_use]
    pub fn reflection_at_port1(&self, omega: Scalar, z0: Scalar, z_load: C) -> C {
        let zin = self.input_impedance(omega, z_load);
        let z0c = C::new(z0, 0.0);
        (zin - z0c) / (zin + z0c)
    }
}

#[cfg(test)]
mod tests_ext {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn input_impedance_reduces_to_load_for_zero_length() {
        let tl = TransmissionLine::lossless(0.0, 250e-9, 100e-12);
        let zl = C::new(75.0, 0.0);
        let zin = tl.input_impedance(2.0 * std::f64::consts::PI * 1.0e9, zl);
        assert_relative_eq!(zin.re, zl.re, epsilon = 1.0e-12);
        assert_relative_eq!(zin.im, zl.im, epsilon = 1.0e-12);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn identity_when_length_zero() {
        let tl = TransmissionLine::lossless(0.0, 1e-6, 1e-9);
        let t = tl.to_twoport(1.0e6);
        assert_relative_eq!(t.a.re, 1.0, epsilon = 1e-12);
        assert_relative_eq!(t.b.norm(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(t.c.norm(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(t.d.re, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn reciprocity_holds_for_uniform_line() {
        let tl = TransmissionLine::lossless(0.1, 250e-9, 100e-12);
        let t = tl.to_twoport(2.0 * std::f64::consts::PI * 1.0e9);
        let det = t.determinant();
        assert_relative_eq!(det.re, 1.0, epsilon = 1e-6);
        assert_relative_eq!(det.im, 0.0, epsilon = 1e-6);
    }
}

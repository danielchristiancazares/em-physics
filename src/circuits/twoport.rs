//! Two-port network representations and parameter conversions.

use num_complex::Complex;

use crate::math::Scalar;

/// Convenience alias for complex scalars.
pub type C = Complex<Scalar>;

/// ABCD-based two-port network.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoPort {
    /// A element of the ABCD matrix.
    pub a: C,
    /// B element of the ABCD matrix.
    pub b: C,
    /// C element of the ABCD matrix.
    pub c: C,
    /// D element of the ABCD matrix.
    pub d: C,
}

impl TwoPort {
    /// Identity two-port (through connection): [[1, 0], [0, 1]].
    #[must_use]
    pub fn identity() -> Self {
        Self {
            a: C::new(1.0, 0.0),
            b: C::new(0.0, 0.0),
            c: C::new(0.0, 0.0),
            d: C::new(1.0, 0.0),
        }
    }

    /// Constructs a two-port from explicit ABCD elements.
    #[must_use]
    pub fn from_abcd(a: C, b: C, c: C, d: C) -> Self {
        Self { a, b, c, d }
    }

    /// Series impedance `Z` represented as a two-port.
    #[must_use]
    pub fn series_impedance(z: C) -> Self {
        Self::from_abcd(C::new(1.0, 0.0), z, C::new(0.0, 0.0), C::new(1.0, 0.0))
    }

    /// Shunt admittance `Y` represented as a two-port.
    #[must_use]
    pub fn shunt_admittance(y: C) -> Self {
        Self::from_abcd(C::new(1.0, 0.0), C::new(0.0, 0.0), y, C::new(1.0, 0.0))
    }

    /// ABCD determinant `ad - bc`.
    #[must_use]
    pub fn determinant(&self) -> C {
        self.a * self.d - self.b * self.c
    }

    /// Cascades this two-port with `rhs` (i.e., self followed by rhs).
    #[must_use]
    pub fn cascade(&self, rhs: &TwoPort) -> TwoPort {
        // Matrix multiplication [[a b],[c d]] * [[a' b'],[c' d']]
        TwoPort {
            a: self.a * rhs.a + self.b * rhs.c,
            b: self.a * rhs.b + self.b * rhs.d,
            c: self.c * rhs.a + self.d * rhs.c,
            d: self.c * rhs.b + self.d * rhs.d,
        }
    }

    /// Converts to Z-parameters when `c != 0`.
    /// Returns `None` if conversion is ill-conditioned.
    #[must_use]
    pub fn to_z(&self) -> Option<[[C; 2]; 2]> {
        if self.c.norm() == 0.0 {
            return None;
        }
        let inv_c = C::new(1.0, 0.0) / self.c;
        let det = self.determinant();
        Some([
            [self.a * inv_c, det * inv_c],
            [inv_c, self.d * inv_c],
        ])
    }

    /// Converts to Y-parameters. Prefers the direct formula for `b != 0`.
    /// Falls back to converting Z and inverting if possible, and recognizes
    /// the ideal shunt case when `b == 0 && a≈1 && d≈1`.
    #[must_use]
    pub fn to_y(&self) -> Option<[[C; 2]; 2]> {
        let eps = 1e-15;
        if self.b.norm() > eps {
            let inv_b = C::new(1.0, 0.0) / self.b;
            let det = self.determinant();
            return Some([
                [self.d * inv_b, -det * inv_b],
                [-inv_b, self.a * inv_b],
            ]);
        }

        // Recognize pure shunt form [[1, 0], [Y, 1]]
        if (self.a - C::new(1.0, 0.0)).norm() <= eps
            && (self.d - C::new(1.0, 0.0)).norm() <= eps
        {
            let y = self.c;
            return Some([[y, -y], [-y, y]]);
        }

        // Fallback: compute Z then invert if non-singular
        if let Some(z) = self.to_z() {
            let det = z[0][0] * z[1][1] - z[0][1] * z[1][0];
            if det.norm() > eps {
                let inv = [
                    [z[1][1] / det, -z[0][1] / det],
                    [-z[1][0] / det, z[0][0] / det],
                ];
                return Some(inv);
            }
        }
        None
    }

    /// Converts to S-parameters for a given real reference impedance `z0`.
    /// Returns `None` if the denominator is zero.
    #[must_use]
    pub fn to_s(&self, z0: Scalar) -> Option<SParameters> {
        let z0c = C::new(z0, 0.0);
        let inv_z0 = C::new(1.0 / z0, 0.0);
        let den = self.a + self.b * inv_z0 + self.c * z0c + self.d;
        if den.norm() == 0.0 {
            return None;
        }
        let num11 = self.a + self.b * inv_z0 - self.c * z0c - self.d;
        let num22 = -self.a + self.b * inv_z0 - self.c * z0c + self.d;
        let s11 = num11 / den;
        let s21 = C::new(2.0, 0.0) / den;
        let s22 = num22 / den;
        let s12 = C::new(2.0, 0.0) * self.determinant() / den;
        Some(SParameters { s11, s12, s21, s22 })
    }

    /// Constructs a `TwoPort` from Z-parameters. Requires `z21 != 0`.
    #[must_use]
    pub fn from_z(z: [[C; 2]; 2]) -> Option<Self> {
        let z11 = z[0][0];
        let z12 = z[0][1];
        let z21 = z[1][0];
        let z22 = z[1][1];
        if z21.norm() == 0.0 {
            return None;
        }
        let det = z11 * z22 - z12 * z21;
        let a = z11 / z21;
        let b = det / z21;
        let c = C::new(1.0, 0.0) / z21;
        let d = z22 / z21;
        Some(Self { a, b, c, d })
    }

    /// Constructs a `TwoPort` from Y-parameters. Requires `y21 != 0`.
    #[must_use]
    pub fn from_y(y: [[C; 2]; 2]) -> Option<Self> {
        let y11 = y[0][0];
        let y12 = y[0][1];
        let y21 = y[1][0];
        let y22 = y[1][1];
        if y21.norm() == 0.0 {
            return None;
        }
        let det = y11 * y22 - y12 * y21;
        let a = -y22 / y21;
        let b = -C::new(1.0, 0.0) / y21;
        let c = -det / y21;
        let d = -y11 / y21;
        Some(Self { a, b, c, d })
    }

    /// Constructs a `TwoPort` from S-parameters using an equal reference impedance `z0`.
    /// Requires `s21 != 0`.
    #[must_use]
    pub fn from_s_equal(s: &SParameters, z0: Scalar) -> Option<Self> {
        if s.s21.norm() == 0.0 {
            return None;
        }
        let one = C::new(1.0, 0.0);
        let z0c = C::new(z0, 0.0);
        let two = C::new(2.0, 0.0);
        let den = two * s.s21;
        let a = ((one + s.s11) * (one - s.s22) + s.s12 * s.s21) / den;
        let b = z0c * ((one + s.s11) * (one + s.s22) - s.s12 * s.s21) / den;
        let c = (one / z0c) * ((one - s.s11) * (one - s.s22) - s.s12 * s.s21) / den;
        let d = ((one - s.s11) * (one + s.s22) + s.s12 * s.s21) / den;
        Some(Self { a, b, c, d })
    }

    /// Converts ABCD to S with possibly unequal real reference impedances per port.
    /// Uses power-wave normalization.
    #[must_use]
    pub fn to_s_unequal(&self, z01: Scalar, z02: Scalar) -> Option<SParameters> {
        let z01c = C::new(z01, 0.0);
        let z02c = C::new(z02, 0.0);
        let den = self.a + self.b / z02c + self.c * z01c + self.d;
        if den.norm() == 0.0 {
            return None;
        }
        let s11 = (self.a + self.b / z02c - self.c * z01c - self.d) / den;
        let s22 = (-self.a + self.b / z02c - self.c * z01c + self.d) / den;
        let s21 = C::new(2.0, 0.0) * (z01 / z02).sqrt() / den;
        let s12 = C::new(2.0, 0.0) * (z02 / z01).sqrt() * self.determinant() / den;
        Some(SParameters { s11, s12, s21, s22 })
    }

    /// Input impedance at port 1 when port 2 is terminated by `z_load`.
    #[must_use]
    pub fn input_impedance(&self, z_load: C) -> C {
        (self.a * z_load + self.b) / (self.c * z_load + self.d)
    }

    /// Reflection coefficient at port 1 for reference impedance `z0_1` and load at port 2.
    #[must_use]
    pub fn reflection_at_port1(&self, z_load: C, z0_1: Scalar) -> C {
        let zin = self.input_impedance(z_load);
        let z0 = C::new(z0_1, 0.0);
        (zin - z0) / (zin + z0)
    }

    /// T-section builder: series `za`, shunt `yb`, series `zc`.
    #[must_use]
    pub fn t_section(za: C, yb: C, zc: C) -> Self {
        Self::series_impedance(za)
            .cascade(&Self::shunt_admittance(yb))
            .cascade(&Self::series_impedance(zc))
    }

    /// Π-section builder: shunt `y1`, series `z2`, shunt `y3`.
    #[must_use]
    pub fn pi_section(y1: C, z2: C, y3: C) -> Self {
        Self::shunt_admittance(y1)
            .cascade(&Self::series_impedance(z2))
            .cascade(&Self::shunt_admittance(y3))
    }

    /// Cascades a sequence of two-ports from first to last. Returns identity for empty.
    #[must_use]
    pub fn cascade_all<'a>(list: impl IntoIterator<Item = &'a TwoPort>) -> TwoPort {
        let mut acc = TwoPort::identity();
        for t in list {
            acc = acc.cascade(t);
        }
        acc
    }

    /// Cascades a sequence of S-parameter blocks with equal reference `z0`.
    /// Converts each to ABCD, cascades, then converts back to S.
    #[must_use]
    pub fn cascade_s_equal<'a>(list: impl IntoIterator<Item = &'a SParameters>, z0: Scalar) -> Option<SParameters> {
        let mut acc = TwoPort::identity();
        for s in list {
            let t = TwoPort::from_s_equal(s, z0)?;
            acc = acc.cascade(&t);
        }
        acc.to_s(z0)
    }
}

/// Scattering parameters under a single real reference impedance.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SParameters {
    /// Reflection at port 1.
    pub s11: C,
    /// Reverse transmission.
    pub s12: C,
    /// Forward transmission.
    pub s21: C,
    /// Reflection at port 2.
    pub s22: C,
}

impl SParameters {
    /// Returns |S21| in dB.
    #[must_use]
    pub fn s21_db(&self) -> Scalar {
        20.0 * self.s21.norm().max(1e-300).log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cascade_identity_is_noop() {
        let t = TwoPort::series_impedance(C::new(50.0, 0.0));
        let id = TwoPort::identity();
        let res = t.cascade(&id);
        assert_relative_eq!(res.a.re, t.a.re, epsilon = 1e-12);
        assert_relative_eq!(res.b.re, t.b.re, epsilon = 1e-12);
        assert_relative_eq!(res.c.re, t.c.re, epsilon = 1e-12);
        assert_relative_eq!(res.d.re, t.d.re, epsilon = 1e-12);
    }

    #[test]
    fn abcd_to_y_for_shunt() {
        let y = C::new(1e-3, 0.0);
        let t = TwoPort::shunt_admittance(y);
        let yparams = t.to_y().expect("convertible");
        // For pure shunt: Y11 = Y22 = Y, Y12 = Y21 = -Y
        assert_relative_eq!(yparams[0][0].re, y.re, epsilon = 1e-12);
        assert_relative_eq!(yparams[1][1].re, y.re, epsilon = 1e-12);
        assert_relative_eq!(yparams[0][1].re, -y.re, epsilon = 1e-12);
        assert_relative_eq!(yparams[1][0].re, -y.re, epsilon = 1e-12);
    }

    #[test]
    fn sparams_of_series_r_match_expectations() {
        let z0 = 50.0;
        let t = TwoPort::series_impedance(C::new(50.0, 0.0));
        let s = t.to_s(z0).unwrap();
        // 50Ω in series with 50Ω reference => |S21| = 2/(2+1) = 2/3
        assert_relative_eq!(s.s21.norm(), 2.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(s.s11.re, 1.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn roundtrip_z_and_y() {
        let z = C::new(10.0, 5.0);
        let y = C::new(1e-3, -2e-3);
        let t1 = TwoPort::series_impedance(z);
        let t2 = TwoPort::shunt_admittance(y);
        let t = t1.cascade(&t2);
        let zparams = t.to_z().unwrap();
        let t_from_z = TwoPort::from_z(zparams).unwrap();
        assert_relative_eq!(t_from_z.a.re, t.a.re, epsilon = 1e-9);
        assert_relative_eq!(t_from_z.b.re, t.b.re, epsilon = 1e-9);
        let yparams = t.to_y().unwrap();
        let t_from_y = TwoPort::from_y(yparams).unwrap();
        assert_relative_eq!(t_from_y.c.re, t.c.re, epsilon = 1e-9);
    }

    #[test]
    fn s_equal_roundtrip() {
        let z0 = 50.0;
        let base = TwoPort::series_impedance(C::new(50.0, 10.0));
        let s = base.to_s(z0).unwrap();
        let back = TwoPort::from_s_equal(&s, z0).unwrap();
        assert_relative_eq!(back.a.re, base.a.re, epsilon = 1e-9);
    }

    #[test]
    fn cascade_of_series_impedances_adds_b_terms() {
        let t1 = TwoPort::series_impedance(C::new(10.0, 1.0));
        let t2 = TwoPort::series_impedance(C::new(5.0, -2.0));
        let tc = t1.cascade(&t2);
        let te = TwoPort::series_impedance(C::new(15.0, -1.0));
        assert_relative_eq!(tc.b.re, te.b.re, epsilon = 1e-12);
        assert_relative_eq!(tc.b.im, te.b.im, epsilon = 1e-12);
        assert_relative_eq!(tc.a.re, 1.0, epsilon = 1e-12);
        assert_relative_eq!(tc.d.re, 1.0, epsilon = 1e-12);
    }
}

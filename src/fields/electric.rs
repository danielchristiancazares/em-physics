use crate::math::{R3, Scalar};

/// Variants describing the physical interpretation of an electric field.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElectricFieldKind {
    /// Static field (ω = 0).
    Static,
    /// Time-harmonic field represented by peak magnitude and angular frequency.
    TimeHarmonic,
    /// Broadband field defined by instantaneous samples.
    Transient,
}

/// Electric field descriptor capturing amplitude and orientation.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct ElectricField {
    /// Peak magnitude in volts per meter (V/m).
    pub magnitude: Scalar,
    /// Unit direction vector; normalized on construction.
    pub direction: R3,
    /// Angular frequency ω in rad/s, when applicable.
    pub angular_frequency: Option<Scalar>,
    /// Descriptor capturing analytic context (static, time-harmonic, transient).
    pub kind: ElectricFieldKind,
}

impl ElectricField {
    /// Constructs a static electric field with the provided magnitude and direction.
    #[must_use]
    pub fn static_field(magnitude: Scalar, direction: R3) -> Self {
        Self::new(magnitude, direction, None, ElectricFieldKind::Static)
    }

    /// Constructs a time-harmonic electric field with angular frequency ω.
    #[must_use]
    pub fn time_harmonic(magnitude: Scalar, direction: R3, angular_frequency: Scalar) -> Self {
        Self::new(
            magnitude,
            direction,
            Some(angular_frequency),
            ElectricFieldKind::TimeHarmonic,
        )
    }

    fn new(
        magnitude: Scalar,
        direction: R3,
        angular_frequency: Option<Scalar>,
        kind: ElectricFieldKind,
    ) -> Self {
        let normalized = direction.normalize();
        Self {
            magnitude,
            direction: normalized,
            angular_frequency,
            kind,
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn static_field_normalizes_direction() {
        let direction = R3::new(0.0, 3.0, 4.0);
        let field = ElectricField::static_field(10.0, direction);
        assert_relative_eq!(field.direction.norm(), 1.0, epsilon = 1.0e-12);
        assert_eq!(field.kind, ElectricFieldKind::Static);
        assert!(field.angular_frequency.is_none());
    }
}

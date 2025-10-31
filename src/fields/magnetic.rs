use crate::math::{R3, Scalar};

/// Magnetic field descriptor expressed in tesla (T).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct MagneticField {
    /// Magnetic flux density magnitude (T).
    pub magnitude: Scalar,
    /// Unit direction vector.
    pub direction: R3,
}

impl MagneticField {
    /// Constructs a magnetic field from a vector representation.
    #[must_use]
    pub fn from_vector(vector: R3) -> Self {
        let magnitude = vector.norm();
        let direction = if magnitude == 0.0 {
            R3::new(0.0, 0.0, 0.0)
        } else {
            vector / magnitude
        };

        Self {
            magnitude,
            direction,
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn from_vector_handles_nonzero_input() {
        let vector = R3::new(1.0, 2.0, 2.0);
        let field = MagneticField::from_vector(vector);
        assert_relative_eq!(field.magnitude, 3.0, epsilon = 1.0e-12);
        assert_relative_eq!(field.direction.norm(), 1.0, epsilon = 1.0e-12);
    }
}

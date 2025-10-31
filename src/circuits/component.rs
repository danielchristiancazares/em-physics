use crate::math::Scalar;
use crate::units::{Farad, Henry, Impedance, Quantity, Volt};
use num_complex::Complex;

/// Trait implemented by all circuit components that can provide a frequency-domain impedance.
pub trait Component {
    /// Returns the component's impedance for an angular frequency `omega` (rad/s).
    fn impedance(&self, omega: Scalar) -> Complex<Scalar>;

    /// Human-readable identifier (e.g. `R1`).
    fn name(&self) -> &str;
}

/// Lumped resistor model.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct Resistor {
    name: String,
    resistance: Impedance<Scalar>,
}

impl Resistor {
    /// Creates a resistor.
    #[must_use]
    pub fn new(name: impl Into<String>, resistance_ohms: Scalar) -> Self {
        Self {
            name: name.into(),
            resistance: Impedance::new(resistance_ohms),
        }
    }

    /// Resistance magnitude in ohms.
    #[must_use]
    pub fn resistance(&self) -> Scalar {
        self.resistance.value()
    }
}

impl Component for Resistor {
    fn impedance(&self, _omega: Scalar) -> Complex<Scalar> {
        Complex::new(self.resistance(), 0.0)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Lumped capacitor model (ideal).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct Capacitor {
    name: String,
    capacitance: Quantity<Scalar, Farad>,
}

impl Capacitor {
    /// Creates a capacitor.
    #[must_use]
    pub fn new(name: impl Into<String>, capacitance_f: Scalar) -> Self {
        Self {
            name: name.into(),
            capacitance: Quantity::new(capacitance_f),
        }
    }

    /// Returns the capacitance magnitude in farads.
    #[must_use]
    pub fn capacitance(&self) -> Scalar {
        self.capacitance.value()
    }
}

impl Component for Capacitor {
    fn impedance(&self, omega: Scalar) -> Complex<Scalar> {
        if omega.abs() < Scalar::EPSILON {
            Complex::new(f64::INFINITY, 0.0)
        } else {
            Complex::new(0.0, -1.0 / (omega * self.capacitance()))
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Lumped inductor model (ideal).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct Inductor {
    name: String,
    inductance: Quantity<Scalar, Henry>,
}

impl Inductor {
    /// Creates an inductor.
    #[must_use]
    pub fn new(name: impl Into<String>, inductance_h: Scalar) -> Self {
        Self {
            name: name.into(),
            inductance: Quantity::new(inductance_h),
        }
    }

    /// Returns the inductance magnitude in henries.
    #[must_use]
    pub fn inductance(&self) -> Scalar {
        self.inductance.value()
    }
}

impl Component for Inductor {
    fn impedance(&self, omega: Scalar) -> Complex<Scalar> {
        Complex::new(0.0, omega * self.inductance())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Ideal voltage source represented by an amplitude and phase.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct VoltageSource {
    /// Externally visible name.
    pub name: String,
    /// RMS voltage magnitude.
    pub voltage: Quantity<Scalar, Volt>,
    /// Phase angle in radians.
    pub phase: Scalar,
}

impl VoltageSource {
    /// Creates a sinusoidal voltage source.
    #[must_use]
    pub fn new(name: impl Into<String>, voltage_v: Scalar, phase: Scalar) -> Self {
        Self {
            name: name.into(),
            voltage: Quantity::new(voltage_v),
            phase,
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn resistor_impedance_is_real() {
        let r = Resistor::new("R1", 100.0);
        let z = r.impedance(1.0);
        assert_relative_eq!(z.re, 100.0);
        assert_relative_eq!(z.im, 0.0);
    }

    #[test]
    fn capacitor_impedance_is_reactive() {
        let c = Capacitor::new("C1", 1e-6);
        let omega = 1.0e3;
        let z = c.impedance(omega);
        assert_relative_eq!(z.re, 0.0, epsilon = 1.0e-12);
        assert!(z.im < 0.0);
    }
}

use std::sync::Arc;

use num_complex::Complex;

use crate::math::Scalar;

use super::component::Component;

/// Connection topology for a collection of components.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionKind {
    /// Series connection (impedances add linearly).
    Series,
    /// Parallel connection (admittances add linearly).
    Parallel,
}

/// Simple aggregate network that groups multiple components with a shared connection style.
pub struct Network {
    name: String,
    connection: ConnectionKind,
    members: Vec<Arc<dyn Component + Send + Sync>>,
}

impl Network {
    /// Creates a new network.
    #[must_use]
    pub fn new(name: impl Into<String>, connection: ConnectionKind) -> Self {
        Self {
            name: name.into(),
            connection,
            members: Vec::new(),
        }
    }

    /// Adds a component to the network.
    pub fn add_component<C>(&mut self, component: C)
    where
        C: Component + Send + Sync + 'static,
    {
        self.members.push(Arc::new(component));
    }

    /// Returns the aggregate impedance for the network.
    #[must_use]
    pub fn impedance(&self, omega: Scalar) -> Complex<Scalar> {
        match self.connection {
            ConnectionKind::Series => {
                let mut total = Complex::<Scalar>::default();
                for component in &self.members {
                    total += component.impedance(omega);
                }
                total
            }
            ConnectionKind::Parallel => {
                let mut admittance = Complex::<Scalar>::default();
                for component in &self.members {
                    let z = component.impedance(omega);
                    if z.norm() <= Scalar::EPSILON {
                        return Complex::new(Scalar::INFINITY, 0.0);
                    }
                    admittance += Complex::new(1.0, 0.0) / z;
                }

                if admittance.norm() <= Scalar::EPSILON {
                    Complex::new(Scalar::INFINITY, 0.0)
                } else {
                    Complex::new(1.0, 0.0) / admittance
                }
            }
        }
    }
}

impl Network {
    /// Returns the name of the network.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the connection topology.
    #[must_use]
    pub fn connection_kind(&self) -> ConnectionKind {
        self.connection
    }

    /// Returns the number of components in the network.
    #[must_use]
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Returns true when no components are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}

impl std::fmt::Debug for Network {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Network")
            .field("name", &self.name)
            .field("connection", &self.connection)
            .field("members", &self.members.len())
            .finish()
    }
}

impl Default for Network {
    fn default() -> Self {
        Self {
            name: String::from("network"),
            connection: ConnectionKind::Series,
            members: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::circuits::component::{Capacitor, Resistor};

    #[test]
    fn series_network_adds_impedances() {
        let mut network = Network::new("series", ConnectionKind::Series);
        network.add_component(Resistor::new("R1", 100.0));
        network.add_component(Resistor::new("R2", 50.0));
        let z = network.impedance(1.0);
        assert_relative_eq!(z.re, 150.0);
    }

    #[test]
    fn parallel_network_combines_admittance() {
        let mut network = Network::new("parallel", ConnectionKind::Parallel);
        network.add_component(Resistor::new("R1", 100.0));
        network.add_component(Resistor::new("R2", 100.0));
        let z = network.impedance(1.0);
        assert_relative_eq!(z.re, 50.0);
    }

    #[test]
    fn parallel_handles_reactive_elements() {
        let mut network = Network::new("parallel", ConnectionKind::Parallel);
        network.add_component(Capacitor::new("C1", 1e-6));
        let z = network.impedance(1.0e3);
        assert!(z.im.abs() > 0.0);
    }
}

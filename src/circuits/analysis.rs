use nalgebra::{DMatrix, DVector};
use num_complex::Complex;

use crate::math::Scalar;
use crate::circuits::stamp::{MnaBuilder, SolveReport};

/// Dense admittance matrix used in nodal analysis.
pub type AdmittanceMatrix = DMatrix<Complex<Scalar>>;
/// Complex current injection vector.
pub type CurrentVector = DVector<Complex<Scalar>>;

/// Helper for constructing and solving nodal analysis systems.
#[derive(Debug, Clone)]
pub struct NodalAnalysis {
    /// System admittance matrix.
    pub admittance: AdmittanceMatrix,
    /// Net current injection vector.
    pub current: CurrentVector,
}

impl NodalAnalysis {
    /// Creates an empty system with `node_count` nodes.
    #[must_use]
    pub fn new(node_count: usize) -> Self {
        let size = node_count;
        Self {
            admittance: AdmittanceMatrix::zeros(size, size),
            current: CurrentVector::zeros(size),
        }
    }

    /// Solves for nodal voltages using LU factorization.
    #[must_use]
    pub fn solve(&self) -> Option<DVector<Complex<Scalar>>> {
        self.admittance.clone().lu().solve(&self.current)
    }
}

/// Result of evaluating a network at a single angular frequency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrequencyPoint {
    /// Angular frequency ω in rad/s.
    pub omega: Scalar,
    /// Equivalent input impedance seen by the source.
    pub impedance: Complex<Scalar>,
}

/// Computes the equivalent impedance of a `Network` across the provided angular
/// frequencies. This is a convenience for quick AC sweeps on aggregate
/// networks without constructing a full nodal system.
#[must_use]
pub fn sweep_network_impedance<I>(
    network: &crate::circuits::network::Network,
    omegas: I,
) -> Vec<FrequencyPoint>
where
    I: IntoIterator<Item = Scalar>,
{
    omegas
        .into_iter()
        .map(|w| FrequencyPoint {
            omega: w,
            impedance: network.impedance(w),
        })
        .collect()
}

/// AC sweep result for an MNA system at a single frequency.
#[derive(Debug, Clone)]
pub struct AcPointMna {
    /// Angular frequency ω (rad/s).
    pub omega: Scalar,
    /// Node voltages (complex phasors) of size `node_count`.
    pub voltages: DVector<Complex<Scalar>>,
    /// Source currents (complex), in the order voltage sources were stamped.
    pub source_currents: DVector<Complex<Scalar>>,
    /// Diagnostics gathered during solve.
    pub report: SolveReport,
}

/// Sweeps an MNA-stamped circuit across `omegas`.
/// The `stamp` closure receives `(omega, &mut MnaBuilder)` and must fully stamp the circuit
/// for that frequency.
#[must_use]
pub fn ac_sweep_mna<I, F>(node_count: usize, omegas: I, mut stamp: F) -> Vec<AcPointMna>
where
    I: IntoIterator<Item = Scalar>,
    F: FnMut(Scalar, &mut MnaBuilder),
{
    let mut out = Vec::new();
    for w in omegas {
        let mut mna = MnaBuilder::new(node_count);
        stamp(w, &mut mna);
        let (x, report) = mna.solve_with_report();
        if let Some(x) = x {
            let (v, i) = mna.split_solution(x);
            out.push(AcPointMna { omega: w, voltages: v, source_currents: i, report });
        } else {
            out.push(AcPointMna {
                omega: w,
                voltages: DVector::zeros(mna.dimensions().0),
                source_currents: DVector::zeros(mna.dimensions().1),
                report,
            });
        }
    }
    out
}

use std::io;
use std::io::Write;

/// Writes `FrequencyPoint` vector to a CSV writer.
pub fn write_frequency_points_csv<W: Write>(mut w: W, points: &[FrequencyPoint]) -> io::Result<()> {
    writeln!(w, "omega,ReZ,ImZ")?;
    for p in points {
        writeln!(w, "{:.16e},{:.16e},{:.16e}", p.omega, p.impedance.re, p.impedance.im)?;
    }
    Ok(())
}

/// Writes a CSV of node voltage at `node_index` across an AC MNA sweep.
pub fn write_ac_points_mna_node_csv<W: Write>(mut w: W, data: &[AcPointMna], node_index: usize) -> io::Result<()> {
    writeln!(w, "omega,ReV,ImV,cond_estimate,success")?;
    for p in data {
        let v = if node_index < p.voltages.len() { p.voltages[node_index] } else { Complex::new(0.0, 0.0) };
        let cond = p.report.cond_estimate.unwrap_or(f64::NAN);
        writeln!(w, "{:.16e},{:.16e},{:.16e},{:.6e},{}", p.omega, v.re, v.im, cond, p.report.success)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::circuits::network::{ConnectionKind, Network};

    #[test]
    fn solve_returns_expected_voltage() {
        let mut system = NodalAnalysis::new(1);
        system.admittance[(0, 0)] = Complex::new(1.0, 0.0);
        system.current[0] = Complex::new(1.0, 0.0);
        let voltages = system.solve().expect("solution exists");
        assert_relative_eq!(voltages[0].re, 1.0, epsilon = 1.0e-12);
    }

    #[test]
    fn sweep_network_impedance_returns_points() {
        let mut net = Network::new("series", ConnectionKind::Series);
        net.add_component(crate::circuits::component::Resistor::new("R", 50.0));
        let data = sweep_network_impedance(&net, [100.0, 1_000.0, 10_000.0]);
        assert_eq!(data.len(), 3);
        for p in data {
            assert_relative_eq!(p.impedance.re, 50.0, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn ac_sweep_mna_rc_norton() {
        // 1 node: Norton source 1A into node; R to GND; C to GND.
        // V = I / (1/R + jωC). Check low/high freq.
        let r = 1_000.0;
        let c = 1e-6;
        let omegas = [0.0, 2.0 * std::f64::consts::PI * 1.0e6];
        let pts = ac_sweep_mna(1, omegas, |w, mna| {
            mna.stamp_current_source(Some(0), None, Complex::new(1.0, 0.0));
            mna.stamp_resistor(Some(0), None, r);
            mna.stamp_capacitor(Some(0), None, c, crate::circuits::stamp::AcContext { omega: w });
        });
        assert_eq!(pts.len(), 2);
        let v_dc = pts[0].voltages[0];
        assert_relative_eq!(v_dc.re, r, epsilon = 1e-9);
        let v_hf = pts[1].voltages[0];
        let mag = v_hf.norm();
        let expected = 1.0 / (omegas[1] * c);
        assert!((mag / expected - 1.0).abs() < 1e-3);
    }
}

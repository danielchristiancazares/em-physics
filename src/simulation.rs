//! High-level orchestration for time and frequency-domain experiments.

use std::time::Duration;

use crate::math::Scalar;

/// Supported simulation domains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimulationDomain {
    /// Frequency-domain / phasor analysis.
    Frequency,
    /// Time-domain transient analysis.
    Time,
}

/// Metadata describing a simulation experiment.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Human-readable identifier.
    pub name: String,
    /// Domain of operation.
    pub domain: SimulationDomain,
    /// Target angular frequency for single-tone analysis.
    pub angular_frequency: Option<Scalar>,
    /// Total runtime for time-domain simulations.
    pub duration: Option<Duration>,
    /// Time step for integration (when applicable).
    pub time_step: Option<Duration>,
}

impl SimulationConfig {
    /// Creates a default frequency-domain configuration.
    #[must_use]
    pub fn frequency(name: impl Into<String>, angular_frequency: Scalar) -> Self {
        Self {
            name: name.into(),
            domain: SimulationDomain::Frequency,
            angular_frequency: Some(angular_frequency),
            duration: None,
            time_step: None,
        }
    }

    /// Creates a time-domain configuration.
    #[must_use]
    pub fn time(name: impl Into<String>, duration: Duration, time_step: Duration) -> Self {
        Self {
            name: name.into(),
            domain: SimulationDomain::Time,
            angular_frequency: None,
            duration: Some(duration),
            time_step: Some(time_step),
        }
    }
}

/// Trait for simulation engines.
pub trait SimulationEngine {
    /// Executes the simulation using the provided configuration.
    fn run(&mut self, config: &SimulationConfig) -> Result<(), SimulationError>;
}

/// Errors that can occur while configuring or executing simulations.
#[derive(Debug, thiserror::Error)]
pub enum SimulationError {
    /// Raised when a required parameter is missing.
    #[error("missing parameter: {0}")]
    MissingParameter(&'static str),
    /// Raised when the configuration is internally inconsistent.
    #[error("configuration error: {0}")]
    InvalidConfig(String),
}

use nalgebra::DVector;
use num_complex::Complex;

use crate::circuits::stamp::{MnaBuilder, Node};

/// Time integrators for transient MNA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeIntegrator {
    /// Second-order trapezoidal rule (Gear(2) trapezoidal).
    Trapezoidal,
    /// Second-order backward differentiation formula.
    Bdf2,
    /// First-order backward Euler (for robustness on stiff starts).
    Bdf1,
}

/// Captured waveforms from a transient simulation (node voltages and source currents).
#[derive(Debug, Clone, Default)]
pub struct TransientWaveform {
    pub times: Vec<Scalar>,
    /// Node voltages per sample (real scalars). Each entry has length = `node_count`.
    pub node_voltages: Vec<DVector<Scalar>>,
    /// Source currents per sample (real scalars). Each entry has length = number of stamped v-sources.
    pub source_currents: Vec<DVector<Scalar>>,
}

impl TransientWaveform {
    /// Total captured samples.
    #[must_use]
    pub fn len(&self) -> usize { self.times.len() }

    /// True if no samples recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool { self.times.is_empty() }
}

#[derive(Debug, Clone)]
struct ResistorElem {
    a: Node,
    b: Node,
    resistance: Scalar,
}

#[derive(Debug, Clone)]
struct CapacitorElem {
    a: Node,
    b: Node,
    capacitance: Scalar,
    // history
    v_prev: Scalar,
    v_prev2: Scalar,
    i_prev: Scalar,
}

#[derive(Debug, Clone)]
struct InductorElem {
    a: Node,
    b: Node,
    inductance: Scalar,
    // history
    v_prev: Scalar,
    i_prev: Scalar,
    i_prev2: Scalar,
}

type SourceFn = Box<dyn Fn(Scalar) -> Scalar + Send + Sync + 'static>;

struct CurrentSourceElem {
    pos: Node,
    neg: Node,
    value_fn: SourceFn,
}

struct VoltageSourceElem {
    pos: Node,
    neg: Node,
    value_fn: SourceFn,
}

/// Turnkey transient engine built on the dense MNA builder with implicit integrators.
pub struct MnaTransientEngine {
    node_count: usize,
    integrator: TimeIntegrator,
    // elements
    resistors: Vec<ResistorElem>,
    capacitors: Vec<CapacitorElem>,
    inductors: Vec<InductorElem>,
    current_sources: Vec<CurrentSourceElem>,
    voltage_sources: Vec<VoltageSourceElem>,
    // capture
    waveform: TransientWaveform,
}

impl MnaTransientEngine {
    /// Creates a transient engine for a circuit with `node_count` non-ground nodes.
    #[must_use]
    pub fn new(node_count: usize, integrator: TimeIntegrator) -> Self {
        Self {
            node_count,
            integrator,
            resistors: Vec::new(),
            capacitors: Vec::new(),
            inductors: Vec::new(),
            current_sources: Vec::new(),
            voltage_sources: Vec::new(),
            waveform: TransientWaveform::default(),
        }
    }

    /// Adds a resistor.
    pub fn add_resistor(&mut self, a: Node, b: Node, resistance: Scalar) {
        self.resistors.push(ResistorElem { a, b, resistance });
    }

    /// Adds a capacitor with optional initial voltage across (a - b).
    pub fn add_capacitor(&mut self, a: Node, b: Node, capacitance: Scalar, initial_voltage: Scalar) {
        self.capacitors.push(CapacitorElem {
            a,
            b,
            capacitance,
            v_prev: initial_voltage,
            v_prev2: initial_voltage,
            i_prev: 0.0,
        });
    }

    /// Adds an inductor with optional initial current from a -> b.
    pub fn add_inductor(&mut self, a: Node, b: Node, inductance: Scalar, initial_current: Scalar) {
        self.inductors.push(InductorElem {
            a,
            b,
            inductance,
            v_prev: 0.0,
            i_prev: initial_current,
            i_prev2: initial_current,
        });
    }

    /// Adds a time-varying current source i(t) from pos to neg.
    pub fn add_current_source<F>(&mut self, pos: Node, neg: Node, value_fn: F)
    where
        F: Fn(Scalar) -> Scalar + Send + Sync + 'static,
    {
        self.current_sources.push(CurrentSourceElem { pos, neg, value_fn: Box::new(value_fn) });
    }

    /// Adds a time-varying voltage source v(t) between pos and neg.
    pub fn add_voltage_source<F>(&mut self, pos: Node, neg: Node, value_fn: F)
    where
        F: Fn(Scalar) -> Scalar + Send + Sync + 'static,
    {
        self.voltage_sources.push(VoltageSourceElem { pos, neg, value_fn: Box::new(value_fn) });
    }

    /// Returns a reference to the captured waveform (populated after `run`).
    #[must_use]
    pub fn waveform(&self) -> &TransientWaveform { &self.waveform }

    /// Consumes and returns the waveform.
    #[must_use]
    pub fn into_waveform(self) -> TransientWaveform { self.waveform }

    fn stamp_passive_elements(&self, mna: &mut MnaBuilder, dt: Scalar) {
        // Linear resistors
        for r in &self.resistors {
            mna.stamp_resistor(r.a, r.b, r.resistance);
        }
        // Cap/Ind are stamped in method-specific pass; this function only handles resistors.
        let _ = dt; // silence unused if optimized away
    }

    fn stamp_dynamic_elements(&self, mna: &mut MnaBuilder, dt: Scalar, time: Scalar) {
        // time-dependent sources
        for s in &self.current_sources {
            let val = (s.value_fn)(time);
            mna.stamp_current_source(s.pos, s.neg, Complex::new(val, 0.0));
        }
        for v in &self.voltage_sources {
            let val = (v.value_fn)(time);
            let _k = mna.stamp_voltage_source(v.pos, v.neg, Complex::new(val, 0.0));
            let _ = _k; // ensure consistent order; not used here directly
        }
    }

    fn stamp_integrator_elements(&self, mna: &mut MnaBuilder, dt: Scalar) {
        match self.integrator {
            TimeIntegrator::Trapezoidal => {
                // Capacitors: G = 2C/dt, I_hist = -G*v_prev - i_prev
                for c in &self.capacitors {
                    let g = 2.0 * c.capacitance / dt;
                    if g > 0.0 {
                        mna.stamp_resistor(c.a, c.b, 1.0 / g);
                    }
                    let i_hist = -g * c.v_prev - c.i_prev;
                    if i_hist != 0.0 {
                        mna.stamp_current_source(c.a, c.b, Complex::new(i_hist, 0.0));
                    }
                }
                // Inductors: G = dt/(2L), I_hist = i_prev + G*v_prev
                for l in &self.inductors {
                    let g = dt / (2.0 * l.inductance);
                    if g > 0.0 {
                        mna.stamp_resistor(l.a, l.b, 1.0 / g);
                    }
                    let i_hist = l.i_prev + g * l.v_prev;
                    if i_hist != 0.0 {
                        mna.stamp_current_source(l.a, l.b, Complex::new(i_hist, 0.0));
                    }
                }
            }
            TimeIntegrator::Bdf1 => {
                // Capacitors: G = C/dt, I_hist = -G*v_prev
                for c in &self.capacitors {
                    let g = c.capacitance / dt;
                    if g > 0.0 {
                        mna.stamp_resistor(c.a, c.b, 1.0 / g);
                    }
                    let i_hist = -g * c.v_prev;
                    if i_hist != 0.0 {
                        mna.stamp_current_source(c.a, c.b, Complex::new(i_hist, 0.0));
                    }
                }
                // Inductors: G = dt/L, I_hist = i_prev
                for l in &self.inductors {
                    let g = dt / l.inductance;
                    if g > 0.0 {
                        mna.stamp_resistor(l.a, l.b, 1.0 / g);
                    }
                    let i_hist = l.i_prev;
                    if i_hist != 0.0 {
                        mna.stamp_current_source(l.a, l.b, Complex::new(i_hist, 0.0));
                    }
                }
            }
            TimeIntegrator::Bdf2 => {
                // Capacitors: i_n = (3/2 C/dt) v_n + C/dt*(-2 v_{n-1} + 1/2 v_{n-2})
                for c in &self.capacitors {
                    let g = 1.5 * c.capacitance / dt;
                    if g > 0.0 {
                        mna.stamp_resistor(c.a, c.b, 1.0 / g);
                    }
                    let i_hist = (c.capacitance / dt) * (-2.0 * c.v_prev + 0.5 * c.v_prev2);
                    if i_hist != 0.0 {
                        mna.stamp_current_source(c.a, c.b, Complex::new(i_hist, 0.0));
                    }
                }
                // Inductors: i_n = (2/3 dt/L) v_n + (4/3) i_{n-1} - (1/3) i_{n-2}
                for l in &self.inductors {
                    let g = (2.0 / 3.0) * (dt / l.inductance);
                    if g > 0.0 {
                        mna.stamp_resistor(l.a, l.b, 1.0 / g);
                    }
                    let i_hist = (4.0 / 3.0) * l.i_prev - (1.0 / 3.0) * l.i_prev2;
                    if i_hist != 0.0 {
                        mna.stamp_current_source(l.a, l.b, Complex::new(i_hist, 0.0));
                    }
                }
            }
        }
    }

    fn update_histories(&mut self, dt: Scalar, solved_voltages: &DVector<Complex<Scalar>>) {
        // Helper to read node potential (real) or 0.0 for ground.
        let mut node_v = |n: Node| -> Scalar {
            match n {
                Some(idx) => solved_voltages[idx].re,
                None => 0.0,
            }
        };

        match self.integrator {
            TimeIntegrator::Trapezoidal => {
                // Capacitor histories
                for c in &mut self.capacitors {
                    let v_now = node_v(c.a) - node_v(c.b);
                    let g = 2.0 * c.capacitance / dt;
                    let i_hist = -g * c.v_prev - c.i_prev;
                    let i_now = g * v_now + i_hist;
                    c.v_prev2 = c.v_prev;
                    c.v_prev = v_now;
                    c.i_prev = i_now;
                }
                // Inductor histories
                for l in &mut self.inductors {
                    let v_now = node_v(l.a) - node_v(l.b);
                    let g = dt / (2.0 * l.inductance);
                    let i_hist = l.i_prev + g * l.v_prev;
                    let i_now = g * v_now + i_hist;
                    l.v_prev = v_now;
                    l.i_prev2 = l.i_prev;
                    l.i_prev = i_now;
                }
            }
            TimeIntegrator::Bdf1 => {
                for c in &mut self.capacitors {
                    let v_now = node_v(c.a) - node_v(c.b);
                    let g = c.capacitance / dt;
                    let i_hist = -g * c.v_prev;
                    let i_now = g * v_now + i_hist;
                    c.v_prev2 = c.v_prev;
                    c.v_prev = v_now;
                    c.i_prev = i_now;
                }
                for l in &mut self.inductors {
                    let v_now = node_v(l.a) - node_v(l.b);
                    let g = dt / l.inductance;
                    let i_hist = l.i_prev;
                    let i_now = g * v_now + i_hist;
                    l.v_prev = v_now;
                    l.i_prev2 = l.i_prev;
                    l.i_prev = i_now;
                }
            }
            TimeIntegrator::Bdf2 => {
                for c in &mut self.capacitors {
                    let v_now = node_v(c.a) - node_v(c.b);
                    let g = 1.5 * c.capacitance / dt;
                    let i_hist = (c.capacitance / dt) * (-2.0 * c.v_prev + 0.5 * c.v_prev2);
                    let i_now = g * v_now + i_hist;
                    c.v_prev2 = c.v_prev;
                    c.v_prev = v_now;
                    c.i_prev = i_now;
                }
                for l in &mut self.inductors {
                    let v_now = node_v(l.a) - node_v(l.b);
                    let g = (2.0 / 3.0) * (dt / l.inductance);
                    let i_hist = (4.0 / 3.0) * l.i_prev - (1.0 / 3.0) * l.i_prev2;
                    let i_now = g * v_now + i_hist;
                    l.v_prev = v_now;
                    l.i_prev2 = l.i_prev;
                    l.i_prev = i_now;
                }
            }
        }
    }
}

impl SimulationEngine for MnaTransientEngine {
    fn run(&mut self, config: &SimulationConfig) -> Result<(), SimulationError> {
        if config.domain != SimulationDomain::Time {
            return Err(SimulationError::InvalidConfig("MnaTransientEngine requires Time domain".into()));
        }
        let duration = config
            .duration
            .ok_or(SimulationError::MissingParameter("duration"))?;
        let time_step = config
            .time_step
            .ok_or(SimulationError::MissingParameter("time_step"))?;
        let total_time = duration.as_secs_f64();
        let dt = time_step.as_secs_f64();
        if dt <= 0.0 {
            return Err(SimulationError::InvalidConfig("time_step must be > 0".into()));
        }

        // number of steps including t=0
        let steps = (total_time / dt).floor() as usize + 1;

        self.waveform.times.clear();
        self.waveform.node_voltages.clear();
        self.waveform.source_currents.clear();

        let mut t = 0.0;
        for _k in 0..steps {
            let mut mna = MnaBuilder::new(self.node_count);

            // Passive linear elements
            self.stamp_passive_elements(&mut mna, dt);

            // Integrator companion models
            self.stamp_integrator_elements(&mut mna, dt);

            // Time-varying sources at current time
            self.stamp_dynamic_elements(&mut mna, dt, t);

            let (x_opt, _report) = mna.solve_with_report();
            let x = x_opt.ok_or_else(|| SimulationError::InvalidConfig("linear solve failed in transient step".into()))?;
            let (v_complex, i_sources_complex) = mna.split_solution(x);

            // Capture waveforms (real parts)
            let v_real = v_complex.map(|c| c.re);
            let i_real = i_sources_complex.map(|c| c.re);
            self.waveform.times.push(t);
            self.waveform.node_voltages.push(v_real.clone());
            self.waveform.source_currents.push(i_real);

            // Update element histories for next step
            self.update_histories(dt, &v_complex);

            t += dt;
        }

        Ok(())
    }
}

use std::io;
use std::io::Write;

/// Writes a CSV of a node's voltage over time from a transient waveform.
pub fn write_transient_node_csv<W: Write>(mut w: W, waveform: &TransientWaveform, node_index: usize) -> io::Result<()> {
    writeln!(w, "time,voltage")?;
    for (idx, time) in waveform.times.iter().enumerate() {
        let v = if node_index < waveform.node_voltages[idx].len() { waveform.node_voltages[idx][node_index] } else { 0.0 };
        writeln!(w, "{:.16e},{:.16e}", time, v)?;
    }
    Ok(())
}

/// Writes a CSV of a voltage-source current over time (index follows stamping order per step).
pub fn write_transient_vsource_current_csv<W: Write>(mut w: W, waveform: &TransientWaveform, source_index: usize) -> io::Result<()> {
    writeln!(w, "time,current")?;
    for (idx, time) in waveform.times.iter().enumerate() {
        let i = if source_index < waveform.source_currents[idx].len() { waveform.source_currents[idx][source_index] } else { 0.0 };
        writeln!(w, "{:.16e},{:.16e}", time, i)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyEngine;

    impl SimulationEngine for DummyEngine {
        fn run(&mut self, config: &SimulationConfig) -> Result<(), SimulationError> {
            if config.domain == SimulationDomain::Frequency && config.angular_frequency.is_none()
            {
                return Err(SimulationError::MissingParameter("angular_frequency"));
            }
            Ok(())
        }
    }

    #[test]
    fn dummy_engine_validates_frequency_configs() {
        let mut engine = DummyEngine;
        let config = SimulationConfig::frequency("freq", 1.0);
        engine.run(&config).expect("valid config");
    }
}

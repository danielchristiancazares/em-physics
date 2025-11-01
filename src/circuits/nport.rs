//! N-port network elements and Touchstone sNp import.
//!
//! Provides utilities to parse Touchstone files and stamp N-port admittance
//! models into MNA systems for AC analysis.

use nalgebra::{DMatrix, DVector};
use num_complex::Complex;

use crate::math::Scalar;

use super::stamp::{AcContext, MnaBuilder, Node};

/// Representation of an N-port network as frequency-dependent S-parameters.
#[derive(Debug, Clone)]
pub struct NPortNetwork {
    pub port_count: usize,
    pub z0: Scalar,
    /// Frequencies in Hz.
    pub frequencies: Vec<Scalar>,
    /// For each frequency index f, an (n x n) matrix of S-parameters.
    pub sparams: Vec<DMatrix<Complex<Scalar>>>,
}

impl NPortNetwork {
    /// Returns the S-parameter matrix interpolated (nearest neighbor) at `freq_hz`.
    fn s_at(&self, freq_hz: Scalar) -> &DMatrix<Complex<Scalar>> {
        if self.frequencies.is_empty() {
            panic!("empty NPortNetwork");
        }
        // Nearest neighbor selection
        let mut best_idx = 0usize;
        let mut best_err = (self.frequencies[0] - freq_hz).abs();
        for (i, f) in self.frequencies.iter().enumerate() {
            let err = (f - freq_hz).abs();
            if err < best_err { best_err = err; best_idx = i; }
        }
        &self.sparams[best_idx]
    }

    /// Convert S-parameters to Y-parameters at a given frequency.
    ///
    /// Y = (I - S) * (I + S)^{-1} * (1/Z0)
    fn s_to_y(&self, s: &DMatrix<Complex<Scalar>>) -> DMatrix<Complex<Scalar>> {
        let n = self.port_count;
        let i = DMatrix::<Complex<Scalar>>::identity(n, n);
        let inv = (i.clone() + s).try_inverse().unwrap_or_else(|| {
            // Fallback: pseudo-inverse via LU if singular
            (i.clone() + s).lu().inverse().unwrap_or(DMatrix::zeros(n, n))
        });
        let factor = Complex::new(1.0 / self.z0, 0.0);
        (i - s) * inv * factor
    }

    /// Stamps the N-port as a frequency-domain port admittance matrix between `ports`.
    ///
    /// Each port i is described by a pair of nodes (p_i, n_i) for positive and negative terminals.
    ///
    /// For Y-port matrix Y (n x n), contributions are applied for each Y_ij as a 2x2 block
    /// between the node pairs of port i and port j.
    pub fn stamp_into_mna(&self, ctx: AcContext, ports: &[(Node, Node)], mna: &mut MnaBuilder) {
        assert_eq!(ports.len(), self.port_count, "port count mismatch");
        if ctx.omega == 0.0 { return; }

        let freq_hz = ctx.omega / (2.0 * std::f64::consts::PI);
        let s = self.s_at(freq_hz);
        let y = self.s_to_y(s);

        // For each Y_ij, stamp 4 entries coupling port voltages V_i and V_j.
        // V_i = V(p_i) - V(n_i). Current defined into the network.
        for i in 0..self.port_count {
            let (pi, ni) = ports[i];
            let pi_idx = pi; let ni_idx = ni;
            for j in 0..self.port_count {
                let (pj, nj) = ports[j];
                let yij = y[(i, j)];
                if yij == Complex::new(0.0, 0.0) { continue; }

                // Stamp 2x2 block:
                // [ +yij  -yij; -yij  +yij ] between (pi,ni) rows and (pj,nj) cols
                if let Some(pi_) = pi_idx { if let Some(pj_) = pj { mna.add_matrix_entry(pi_, pj_,  yij); } }
                if let Some(pi_) = pi_idx { if let Some(nj_) = nj { mna.add_matrix_entry(pi_, nj_, -yij); } }
                if let Some(ni_) = ni_idx { if let Some(pj_) = pj { mna.add_matrix_entry(ni_, pj_, -yij); } }
                if let Some(ni_) = ni_idx { if let Some(nj_) = nj { mna.add_matrix_entry(ni_, nj_,  yij); } }
            }
        }
    }
}

/// Minimal Touchstone sNp parser for common cases: `# Hz S RI R 50` or `# Hz S MA R 50`.
pub fn read_touchstone(contents: &str) -> Result<NPortNetwork, String> {
    let mut z0 = 50.0;
    let mut format = String::from("RI"); // RI, MA, or DB
    let mut freq_unit = String::from("Hz");
    let mut data: Vec<(Scalar, Vec<Complex<Scalar>>)> = Vec::new();
    let mut nports: Option<usize> = None;

    for line in contents.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('!') { continue; }
        if l.starts_with('#') {
            // Example: # Hz S RI R 50
            let tokens: Vec<_> = l[1..].split_whitespace().collect();
            if tokens.len() >= 2 { freq_unit = tokens[0].to_string(); }
            if tokens.len() >= 2 { /* tokens[1] should be 'S' */ }
            if tokens.len() >= 3 { format = tokens[2].to_string(); }
            if let Some(r_pos) = tokens.iter().position(|t| *t == "R") {
                if r_pos + 1 < tokens.len() {
                    if let Ok(v) = tokens[r_pos + 1].parse::<Scalar>() { z0 = v; }
                }
            }
            continue;
        }

        // Data line: f  S11  S21  S12  S22 ... order is standard Touchstone row-wise by port pairs
        let toks: Vec<&str> = l.split_whitespace().collect();
        if toks.is_empty() { continue; }
        let f_parsed: Scalar = toks[0].parse().map_err(|_| "invalid frequency")?;
        let f_hz = match freq_unit.as_str() {
            "Hz" => f_parsed,
            "kHz" => f_parsed * 1e3,
            "MHz" => f_parsed * 1e6,
            "GHz" => f_parsed * 1e9,
            _ => f_parsed,
        };

        // Infer n from number of remaining tokens: per S-parameter we have 2 numbers
        let param_tokens = &toks[1..];
        if param_tokens.is_empty() { continue; }
        let pairs = param_tokens.len() / 2;
        let n = (pairs as f64).sqrt() as usize;
        if n * n * 2 != param_tokens.len() { return Err("malformed sNp row".into()); }
        if nports.is_none() { nports = Some(n); }
        if Some(n) != nports { return Err("inconsistent port count across rows".into()); }

        let mut vals: Vec<Complex<Scalar>> = Vec::with_capacity(n * n);
        for k in 0..pairs {
            let a: Scalar = param_tokens[2 * k].parse().map_err(|_| "invalid parameter")?;
            let b: Scalar = param_tokens[2 * k + 1].parse().map_err(|_| "invalid parameter")?;
            let c = match format.as_str() {
                "RI" => Complex::new(a, b),
                "MA" => {
                    // magnitude, angle (degrees)
                    let mag = a; let ang = b.to_radians();
                    Complex::from_polar(mag, ang)
                }
                "DB" => {
                    // dB, angle (degrees)
                    let mag = 10f64.powf(a / 20.0); let ang = b.to_radians();
                    Complex::from_polar(mag, ang)
                }
                _ => return Err("unsupported Touchstone format".into()),
            };
            vals.push(c);
        }
        data.push((f_hz, vals));
    }

    let n = nports.ok_or_else(|| "missing data".to_string())?;
    let mut freqs = Vec::with_capacity(data.len());
    let mut mats = Vec::with_capacity(data.len());
    for (f, vals) in data.into_iter() {
        freqs.push(f);
        let mut m = DMatrix::zeros(n, n);
        for i in 0..n { for j in 0..n { m[(i, j)] = vals[i * n + j]; } }
        mats.push(m);
    }

    Ok(NPortNetwork { port_count: n, z0, frequencies: freqs, sparams: mats })
}



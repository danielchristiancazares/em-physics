//! Minimal SPICE netlist importer for linear elements (R, C, L, V, I).
//!
//! Supports a practical subset sufficient for basic AC and transient stamping.

use std::collections::HashMap;

use num_complex::Complex;

use crate::math::Scalar;

use super::stamp::{AcContext, MnaBuilder, Node};

#[derive(Debug, Clone)]
pub enum Element {
    R { n1: String, n2: String, value: Scalar },
    C { n1: String, n2: String, value: Scalar },
    L { n1: String, n2: String, value: Scalar },
    V { n1: String, n2: String, dc: Option<Scalar>, ac: Option<Complex<Scalar>> },
    I { n1: String, n2: String, dc: Option<Scalar>, ac: Option<Complex<Scalar>> },
}

#[derive(Debug, Clone)]
pub struct SpiceCircuit {
    pub elements: Vec<Element>,
}

impl SpiceCircuit {
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut elements = Vec::new();
        for line in text.lines() {
            let l = line.trim();
            if l.is_empty() || l.starts_with('*') || l.starts_with(';') { continue; }
            let toks: Vec<&str> = l.split_whitespace().collect();
            if toks.is_empty() { continue; }
            let head = toks[0].to_ascii_uppercase();
            let c = head.chars().next().unwrap_or(' ');
            match c {
                'R' => {
                    if toks.len() < 4 { return Err("Invalid resistor line".into()); }
                    let n1 = toks[1].to_string();
                    let n2 = toks[2].to_string();
                    let val: Scalar = parse_scalar(toks[3])?;
                    elements.push(Element::R { n1, n2, value: val });
                }
                'C' => {
                    if toks.len() < 4 { return Err("Invalid capacitor line".into()); }
                    let n1 = toks[1].to_string();
                    let n2 = toks[2].to_string();
                    let val: Scalar = parse_scalar(toks[3])?;
                    elements.push(Element::C { n1, n2, value: val });
                }
                'L' => {
                    if toks.len() < 4 { return Err("Invalid inductor line".into()); }
                    let n1 = toks[1].to_string();
                    let n2 = toks[2].to_string();
                    let val: Scalar = parse_scalar(toks[3])?;
                    elements.push(Element::L { n1, n2, value: val });
                }
                'V' => {
                    if toks.len() < 4 { return Err("Invalid voltage source line".into()); }
                    let n1 = toks[1].to_string();
                    let n2 = toks[2].to_string();
                    let mut dc = None;
                    let mut ac = None;
                    let mut i = 3;
                    while i < toks.len() {
                        let t = toks[i].to_ascii_uppercase();
                        if t == "DC" && i + 1 < toks.len() { dc = Some(parse_scalar(toks[i + 1])?); i += 2; }
                        else if t == "AC" && i + 2 < toks.len() {
                            let mag = parse_scalar(toks[i + 1])?;
                            let phase_deg = parse_scalar(toks[i + 2])?;
                            ac = Some(Complex::from_polar(mag, phase_deg.to_radians()));
                            i += 3;
                        } else { i += 1; }
                    }
                    elements.push(Element::V { n1, n2, dc, ac });
                }
                'I' => {
                    if toks.len() < 4 { return Err("Invalid current source line".into()); }
                    let n1 = toks[1].to_string();
                    let n2 = toks[2].to_string();
                    let mut dc = None;
                    let mut ac = None;
                    let mut i = 3;
                    while i < toks.len() {
                        let t = toks[i].to_ascii_uppercase();
                        if t == "DC" && i + 1 < toks.len() { dc = Some(parse_scalar(toks[i + 1])?); i += 2; }
                        else if t == "AC" && i + 2 < toks.len() {
                            let mag = parse_scalar(toks[i + 1])?;
                            let phase_deg = parse_scalar(toks[i + 2])?;
                            ac = Some(Complex::from_polar(mag, phase_deg.to_radians()));
                            i += 3;
                        } else { i += 1; }
                    }
                    elements.push(Element::I { n1, n2, dc, ac });
                }
                _ => { /* ignore unsupported cards for now */ }
            }
        }
        Ok(Self { elements })
    }

    /// Stamps the circuit for AC analysis at the given angular frequency.
    pub fn stamp_ac(&self, ctx: AcContext, mna: &mut MnaBuilder) {
        // Map SPICE node names to node indices
        let mut node_map: HashMap<String, Node> = HashMap::new();
        let mut get_node = |name: &str| -> Node {
            if name == "0" || name.eq_ignore_ascii_case("gnd") {
                return None;
            }
            if let Some(n) = node_map.get(name) { return *n; }
            let idx = Some(node_map.len());
            node_map.insert(name.to_string(), idx);
            idx
        };

        for e in &self.elements {
            match e {
                Element::R { n1, n2, value } => {
                    mna.stamp_resistor(get_node(n1), get_node(n2), *value);
                }
                Element::C { n1, n2, value } => {
                    mna.stamp_capacitor(get_node(n1), get_node(n2), *value, ctx);
                }
                Element::L { n1, n2, value } => {
                    mna.stamp_inductor(get_node(n1), get_node(n2), *value, ctx);
                }
                Element::V { n1, n2, dc: _, ac } => {
                    let val = ac.unwrap_or(Complex::new(0.0, 0.0));
                    let _k = mna.stamp_voltage_source(get_node(n1), get_node(n2), val);
                    let _ = _k;
                }
                Element::I { n1, n2, dc: _, ac } => {
                    let val = ac.unwrap_or(Complex::new(0.0, 0.0));
                    mna.stamp_current_source(get_node(n1), get_node(n2), val);
                }
            }
        }
    }
}

fn parse_scalar(tok: &str) -> Result<Scalar, String> {
    // Support suffixes: k, m, u, n, p, g
    let mut s = tok.trim().to_string();
    let (mult, base) = match s.chars().last() {
        Some('k') | Some('K') => (1e3, &s[..s.len()-1]),
        Some('m') => (1e-3, &s[..s.len()-1]),
        Some('u') | Some('U') => (1e-6, &s[..s.len()-1]),
        Some('n') | Some('N') => (1e-9, &s[..s.len()-1]),
        Some('p') | Some('P') => (1e-12, &s[..s.len()-1]),
        Some('g') | Some('G') => (1e9, &s[..s.len()-1]),
        _ => (1.0, &s[..])
    };
    base.parse::<Scalar>().map(|v| v * mult).map_err(|_| "invalid number".into())
}



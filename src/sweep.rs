//! Frequency sweep utilities and post-processing helpers.

use num_complex::Complex;

use crate::constants::angular_frequency;
use crate::math::Scalar;

/// Generates `n` linearly spaced samples in [start, stop].
#[must_use]
pub fn linspace(start: Scalar, stop: Scalar, n: usize) -> Vec<Scalar> {
    match n {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let step = (stop - start) / (n as Scalar - 1.0);
            (0..n).map(|i| start + step * i as Scalar).collect()
        }
    }
}

/// Generates `n` logarithmically spaced samples between `start` and `stop` (Hz).
/// Requires start > 0 and stop > 0.
#[must_use]
pub fn logspace_hz(start_hz: Scalar, stop_hz: Scalar, n: usize) -> Vec<Scalar> {
    assert!(start_hz > 0.0 && stop_hz > 0.0);
    match n {
        0 => Vec::new(),
        1 => vec![start_hz],
        _ => {
            let log_start = start_hz.log10();
            let log_stop = stop_hz.log10();
            let step = (log_stop - log_start) / (n as Scalar - 1.0);
            (0..n)
                .map(|i| 10f64.powf(log_start + step * i as Scalar))
                .collect()
        }
    }
}

/// Angular frequency sweep with linear spacing between f_start and f_stop (Hz).
#[must_use]
pub fn angular_freq_linspace(f_start_hz: Scalar, f_stop_hz: Scalar, n: usize) -> Vec<Scalar> {
    linspace(f_start_hz, f_stop_hz, n)
        .into_iter()
        .map(angular_frequency)
        .collect()
}

/// Angular frequency sweep with logarithmic spacing between f_start and f_stop (Hz).
#[must_use]
pub fn angular_freq_logspace(f_start_hz: Scalar, f_stop_hz: Scalar, n: usize) -> Vec<Scalar> {
    logspace_hz(f_start_hz, f_stop_hz, n)
        .into_iter()
        .map(angular_frequency)
        .collect()
}

/// Applies `f` to each angular frequency and collects results.
#[must_use]
pub fn sweep_map<I, F, T>(omegas: I, mut f: F) -> Vec<T>
where
    I: IntoIterator<Item = Scalar>,
    F: FnMut(Scalar) -> T,
{
    omegas.into_iter().map(|w| f(w)).collect()
}

/// Magnitude of complex sequence.
#[must_use]
pub fn mag(values: impl IntoIterator<Item = Complex<Scalar>>) -> Vec<Scalar> {
    values.into_iter().map(|v| v.norm()).collect()
}

/// Magnitude in dB (20*log10(|x|)), clamping very small values.
#[must_use]
pub fn mag_db(values: impl IntoIterator<Item = Complex<Scalar>>) -> Vec<Scalar> {
    const MIN: Scalar = 1e-300;
    values
        .into_iter()
        .map(|v| 20.0 * (v.norm().max(MIN)).log10())
        .collect()
}

/// Phase in radians of complex sequence.
#[must_use]
pub fn phase_rad(values: impl IntoIterator<Item = Complex<Scalar>>) -> Vec<Scalar> {
    values.into_iter().map(|v| v.arg()).collect()
}

/// Phase in degrees of complex sequence.
#[must_use]
pub fn phase_deg(values: impl IntoIterator<Item = Complex<Scalar>>) -> Vec<Scalar> {
    phase_rad(values).into_iter().map(|r| r.to_degrees()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn linspace_basic() {
        let v = linspace(0.0, 1.0, 5);
        assert_eq!(v, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn mag_phase_roundtrip() {
        let x = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)];
        let m = mag(x.clone());
        let p = phase_deg(x);
        assert_relative_eq!(m[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(m[1], 1.0, epsilon = 1e-12);
        assert_relative_eq!(p[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(p[1], 90.0, epsilon = 1e-12);
    }

    #[test]
    fn sweep_map_runs_function() {
        let ws = vec![1.0, 2.0, 3.0];
        let out = sweep_map(ws, |w| w * 2.0);
        assert_eq!(out, vec![2.0, 4.0, 6.0]);
    }
}

use em_physics::circuits::network::ConnectionKind;
use em_physics::circuits::{component::*, network::Network};
use em_physics::constants::angular_frequency;

fn main() {
    // Series RLC network example.
    let mut net = Network::new("series_rlc", ConnectionKind::Series);
    net.add_component(Resistor::new("R1", 50.0)); // 50 Ω
    net.add_component(Inductor::new("L1", 1e-6)); // 1 µH
    net.add_component(Capacitor::new("C1", 1e-9)); // 1 nF

    // Sweep 1 MHz .. 100 MHz (log-spaced-ish quick demo)
    let freqs = [
        1.0e6_f64, 2.0e6, 5.0e6, 1.0e7, 2.0e7, 5.0e7, 1.0e8,
    ];
    let omegas = freqs.into_iter().map(angular_frequency);

    let data = em_physics::circuits::analysis::sweep_network_impedance(&net, omegas);

    println!("omega(rad/s), Z_real(ohm), Z_imag(ohm)");
    for p in data {
        println!("{:.6e}, {:.6e}, {:.6e}", p.omega, p.impedance.re, p.impedance.im);
    }
}


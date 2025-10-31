use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use em_physics::circuits::analysis::sweep_network_impedance;
use em_physics::circuits::network::ConnectionKind;
use em_physics::circuits::{component::*, network::Network};
use em_physics::constants::angular_frequency;

fn build_series_rlc() -> Network {
    let mut net = Network::new("series_rlc", ConnectionKind::Series);
    net.add_component(Resistor::new("R", 50.0));
    net.add_component(Inductor::new("L", 1e-6));
    net.add_component(Capacitor::new("C", 1e-9));
    net
}

fn bench_network_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("network_sweep");
    let freqs: Vec<f64> = (0..10_000).map(|i| 1.0e6 + i as f64 * 1.0e3).collect();
    let omegas: Vec<f64> = freqs.into_iter().map(angular_frequency).collect();

    group.bench_function(BenchmarkId::new("series_rlc", omegas.len()), |b| {
        b.iter_batched(
            build_series_rlc,
            |net| {
                let _ = sweep_network_impedance(&net, omegas.iter().copied());
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group!(benches, bench_network_sweep);
criterion_main!(benches);


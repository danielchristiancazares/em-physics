use bevy::prelude::*;
use em_physics::circuits::stamp::MnaBuilder;
use em_physics::fields::{
    electric_field_from_vector_potential, magnetic_field_from_lines, vector_potential_from_lines,
    LineCurrent, WireSegment3D,
};
use em_physics::math::{CScalar, R3, Scalar};
use num_complex::Complex;

const BRANCH_COUNT: usize = 3;
const BRANCH_X: [f32; BRANCH_COUNT] = [-220.0, 0.0, 220.0];
const BUS_Y: f32 = 150.0;
const LOAD_Y: f32 = -150.0;
const GATE_Y: f32 = 60.0;
const OBS_POINT: R3 = R3::new(0.0, 0.0, 80.0);
const SUPPLY_VOLTAGE: Scalar = 12.0;
const AC_FREQUENCY: Scalar = 60.0;

#[derive(Resource)]
struct CircuitState {
    gates: [bool; BRANCH_COUNT],
    resistances: [Scalar; BRANCH_COUNT],
    branch_currents: [Scalar; BRANCH_COUNT],
    source_current: Scalar,
    node_voltage: Scalar,
    power: Scalar,
    b_field_mag: Scalar,
    e_field_mag: Scalar,
    dirty: bool,
}

impl Default for CircuitState {
    fn default() -> Self {
        Self {
            gates: [true, false, true],
            resistances: [100.0, 220.0, 330.0],
            branch_currents: [0.0; BRANCH_COUNT],
            source_current: 0.0,
            node_voltage: 0.0,
            power: 0.0,
            b_field_mag: 0.0,
            e_field_mag: 0.0,
            dirty: true,
        }
    }
}

#[derive(Component)]
struct GateDisplay(usize);

#[derive(Component)]
struct BranchDisplay(usize);

#[derive(Component)]
struct InfoText;

pub fn main() {
    App::new()
        .insert_resource(CircuitState::default())
        .add_plugins(
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "em-physics: Interactive Circuit & Field Demo".into(),
                    resolution: (1200.0, 720.0).into(),
                    present_mode: bevy::window::PresentMode::AutoVsync,
                    ..default()
                }),
                ..default()
            }),
        )
        .add_systems(Startup, setup)
        .add_systems(Update, (handle_input, recompute_circuit, update_visuals, update_text))
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());

    // Bus line
    commands.spawn(SpriteBundle {
        sprite: Sprite {
            color: Color::rgb(0.1, 0.1, 0.1),
            custom_size: Some(Vec2::new(640.0, 6.0)),
            ..default()
        },
        transform: Transform::from_translation(Vec3::new(0.0, BUS_Y, 0.0)),
        ..default()
    });

    // Ground line
    commands.spawn(SpriteBundle {
        sprite: Sprite {
            color: Color::rgb(0.05, 0.05, 0.05),
            custom_size: Some(Vec2::new(640.0, 6.0)),
            ..default()
        },
        transform: Transform::from_translation(Vec3::new(0.0, LOAD_Y - 20.0, 0.0)),
        ..default()
    });

    // Observation point indicator
    commands.spawn(SpriteBundle {
        sprite: Sprite {
            color: Color::rgb(0.9, 0.6, 0.1),
            custom_size: Some(Vec2::new(16.0, 16.0)),
            ..default()
        },
        transform: Transform::from_translation(Vec3::new(OBS_POINT.x as f32, OBS_POINT.y as f32, 0.0)),
        ..default()
    });

    // Branch visuals
    for (idx, &x) in BRANCH_X.iter().enumerate() {
        // Vertical conductor
        commands.spawn((
            BranchDisplay(idx),
            SpriteBundle {
                sprite: Sprite {
                    color: Color::rgb(0.2, 0.3, 0.6),
                    custom_size: Some(Vec2::new(12.0, BUS_Y - LOAD_Y - 40.0)),
                    ..default()
                },
                transform: Transform::from_translation(Vec3::new(x, (BUS_Y + LOAD_Y) * 0.5 - 20.0, 0.0)),
                ..default()
            },
        ));

        // Gate rectangle
        commands.spawn((
            GateDisplay(idx),
            SpriteBundle {
                sprite: Sprite {
                    color: Color::rgb(0.2, 0.8, 0.3),
                    custom_size: Some(Vec2::new(48.0, 28.0)),
                    ..default()
                },
                transform: Transform::from_translation(Vec3::new(x, GATE_Y, 1.0)),
                ..default()
            },
        ));

        // Termination block
        commands.spawn(SpriteBundle {
            sprite: Sprite {
                color: Color::rgb(0.3, 0.3, 0.3),
                custom_size: Some(Vec2::new(64.0, 32.0)),
                ..default()
            },
            transform: Transform::from_translation(Vec3::new(x, LOAD_Y, 0.0)),
            ..default()
        });
    }

    // Text overlay
    commands.spawn((
        InfoText,
        TextBundle::from_sections([
            TextSection::new(
                "",
                TextStyle {
                    font_size: 24.0,
                    color: Color::WHITE,
                    ..default()
                },
            ),
        ])
        .with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(16.0),
            left: Val::Px(16.0),
            ..default()
        }),
    ));
}

fn handle_input(mut state: ResMut<CircuitState>, keys: Res<ButtonInput<KeyCode>>) {
    for (i, key) in [KeyCode::Digit1, KeyCode::Digit2, KeyCode::Digit3].iter().enumerate() {
        if keys.just_pressed(*key) {
            state.gates[i] = !state.gates[i];
            state.dirty = true;
        }
    }
    if keys.just_pressed(KeyCode::Space) {
        state.gates = state.gates.map(|g| !g);
        state.dirty = true;
    }
}

fn recompute_circuit(mut state: ResMut<CircuitState>) {
    if !state.dirty {
        return;
    }

    let mut mna = MnaBuilder::new(1);
    mna.stamp_voltage_source(Some(0), None, Complex::new(SUPPLY_VOLTAGE, 0.0));

    for (gate, &res) in state.gates.iter().zip(state.resistances.iter()) {
        if *gate {
            mna.stamp_resistor(Some(0), None, res);
        }
    }

    let solution = match mna.solve() {
        Some(sol) => sol,
        None => {
            state.branch_currents.fill(0.0);
            state.source_current = 0.0;
            state.node_voltage = 0.0;
            state.power = 0.0;
            state.b_field_mag = 0.0;
            state.e_field_mag = 0.0;
            state.dirty = false;
            return;
        }
    };

    let (v_nodes, source_currents) = mna.split_solution(solution);
    let v0 = v_nodes[0].re;
    state.node_voltage = v0;
    for i in 0..BRANCH_COUNT {
        state.branch_currents[i] = if state.gates[i] {
            v0 / state.resistances[i]
        } else {
            0.0
        };
    }
    state.source_current = source_currents.get(0).map(|c| c.re).unwrap_or(0.0);
    state.power = state.node_voltage * state.source_current;

    // Field estimates using line-current model
    let mut lines: Vec<LineCurrent> = Vec::new();
    for i in 0..BRANCH_COUNT {
        if state.branch_currents[i].abs() <= 1e-9 {
            continue;
        }
        let x = BRANCH_X[i] as Scalar;
        let segment = WireSegment3D {
            start: R3::new(x, BUS_Y as Scalar, 0.0),
            end: R3::new(x, LOAD_Y as Scalar, 0.0),
        };
        lines.push(LineCurrent {
            segment,
            current: CScalar::new(state.branch_currents[i], 0.0),
        });
    }

    if lines.is_empty() {
        state.b_field_mag = 0.0;
        state.e_field_mag = 0.0;
    } else {
        let b_vec = magnetic_field_from_lines(OBS_POINT, &lines);
        let b_mag = (b_vec[0].norm_sqr() + b_vec[1].norm_sqr() + b_vec[2].norm_sqr()).sqrt();
        let a_vec = vector_potential_from_lines(OBS_POINT, &lines);
        let omega = 2.0 * std::f64::consts::PI * AC_FREQUENCY;
        let e_vec = electric_field_from_vector_potential(a_vec, omega);
        let e_mag = (e_vec[0].norm_sqr() + e_vec[1].norm_sqr() + e_vec[2].norm_sqr()).sqrt();
        state.b_field_mag = b_mag;
        state.e_field_mag = e_mag;
    }

    state.dirty = false;
}

fn update_visuals(
    state: Res<CircuitState>,
    mut gate_query: Query<(&GateDisplay, &mut Sprite)>,
    mut branch_query: Query<(&BranchDisplay, &mut Sprite)>,
) {
    for (GateDisplay(idx), mut sprite) in gate_query.iter_mut() {
        let color = if state.gates[*idx] {
            Color::rgb(0.1, 0.75, 0.3)
        } else {
            Color::rgb(0.25, 0.25, 0.25)
        };
        sprite.color = color;
    }

    let max_current = state
        .branch_currents
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max)
        .max(0.1);

    for (BranchDisplay(idx), mut sprite) in branch_query.iter_mut() {
        let current = state.branch_currents[*idx].abs();
        let intensity = (current / max_current).clamp(0.0, 1.0) as f32;
        let base = 0.25 + 0.6 * intensity;
        sprite.color = Color::rgb(0.2, base, 1.0 - 0.5 * intensity);
    }
}

fn update_text(state: Res<CircuitState>, mut query: Query<&mut Text, With<InfoText>>) {
    if !state.is_changed() {
        return;
    }

    let mut text = query.single_mut();
    let gate_labels: Vec<String> = state
        .gates
        .iter()
        .enumerate()
        .map(|(i, g)| format!("{}:{}", i + 1, if *g { "ON" } else { "OFF" }))
        .collect();

    let branch_lines: Vec<String> = state
        .branch_currents
        .iter()
        .zip(state.resistances.iter())
        .enumerate()
        .map(|(i, (i_curr, r))| {
            format!(
                "Branch {}: {:.3} A through {:.0} Ω",
                i + 1,
                i_curr,
                r
            )
        })
        .collect();

    text.sections[0].value = format!(
        "Gate toggles: [{}]  (press 1/2/3, space to invert)\n\
Node voltage: {:.3} V   Source current: {:.3} A   Power: {:.3} W\n{}\n|B| at probe: {:.3} μT   |E| (60 Hz): {:.3} mV/m",
        gate_labels.join("  "),
        state.node_voltage,
        state.source_current,
        state.power,
        branch_lines.join("\n"),
        state.b_field_mag * 1e6,
        state.e_field_mag * 1e3,
    );
}


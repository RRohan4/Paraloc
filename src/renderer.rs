use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::window::WindowResolution;
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    config::Config,
    filter::ParticleFilter,
    map::Map,
    particle::{MotionDelta, NoiseParams, Particle},
    robot::Robot,
    sensor_model::SequentialSensorModel,
};

const WINDOW_W: f32 = 1280.0;
const WINDOW_H: f32 = 720.0;

#[derive(Resource)]
struct AppMap(Map);

#[derive(Resource)]
struct AppRobot(Robot);

#[derive(Resource)]
struct AppFilter(ParticleFilter);

#[derive(Resource)]
struct AppRng(StdRng);

#[derive(Resource)]
struct AppConfig(Config);

#[derive(Resource, Default)]
struct AppLastScan(Vec<f32>);

#[derive(Resource)]
struct Layout {
    tile: f32,
    offset: Vec2,
}

#[derive(Component)]
struct ParticleDot;

#[derive(Component)]
struct RobotMarker;

#[derive(Component)]
struct EstimateMarker;

#[derive(Component)]
struct LidarBeam(usize);

pub struct RendererPlugin;

impl Plugin for RendererPlugin {
    fn build(&self, app: &mut App) {
        let config = Config::load_or_default("config.toml");
        let map = Map::from_path(&config.map_path);
        let mut rng = StdRng::seed_from_u64(0);

        let (mw_u, mh_u) = map.dimensions();
        let mw = mw_u as f32;
        let mh = mh_u as f32;

        let particles: Vec<Particle> = (0..config.particle_count)
            .map(|_| loop {
                let x = rng.gen_range(0.0..mw);
                let y = rng.gen_range(0.0..mh);
                if !map.is_wall(x, y) {
                    return Particle {
                        x,
                        y,
                        theta: rng.gen_range(0.0..std::f32::consts::TAU),
                        weight: 1.0 / config.particle_count as f32,
                    };
                }
            })
            .collect();

        let model = Box::new(SequentialSensorModel {
            sigma: config.sensor_sigma,
            n_rays: config.n_rays,
        });
        let noise = NoiseParams {
            translation_sigma: config.translation_sigma,
            rotation_sigma: config.rotation_sigma,
        };
        let filter = ParticleFilter::new(particles, model, noise);
        let robot = Robot::new(config.robot_spawn_x, config.robot_spawn_y, 0.0);

        let tile = (WINDOW_W / mw).min(WINDOW_H / mh) * 0.95;
        let layout = Layout {
            tile,
            offset: Vec2::new(-mw * tile / 2.0, mh * tile / 2.0),
        };

        app.insert_resource(AppMap(map))
            .insert_resource(AppRobot(robot))
            .insert_resource(AppFilter(filter))
            .insert_resource(AppRng(rng))
            .insert_resource(AppConfig(config))
            .insert_resource(AppLastScan::default())
            .insert_resource(layout)
            .add_plugins(DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    resolution: WindowResolution::new(WINDOW_W, WINDOW_H)
                        .with_scale_factor_override(1.0),
                    title: "Paraloc — yellow=truth, green=estimate, WASD/arrows".into(),
                    ..default()
                }),
                ..default()
            }))
            .insert_resource(ClearColor(Color::srgb(0.04, 0.05, 0.08)))
            .add_systems(Startup, setup)
            .add_systems(
                Update,
                (
                    game_loop,
                    sync_lidar_beams,
                    sync_particles,
                    sync_robot_marker,
                    sync_estimate_marker,
                )
                    .chain(),
            );
    }
}

fn world_to_screen(p: Vec2, layout: &Layout) -> Vec2 {
    Vec2::new(
        layout.offset.x + p.x * layout.tile,
        layout.offset.y - p.y * layout.tile,
    )
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    map: Res<AppMap>,
    config: Res<AppConfig>,
    layout: Res<Layout>,
) {
    commands.spawn(Camera2d);

    let (mw, mh) = map.0.dimensions();
    for y in 0..mh {
        for x in 0..mw {
            let is_wall = map.0.is_wall(x as f32, y as f32);
            let color = if is_wall {
                Color::srgb(0.22, 0.27, 0.40)
            } else {
                Color::srgb(0.07, 0.09, 0.13)
            };
            let pos = world_to_screen(
                Vec2::new(x as f32 + 0.5, y as f32 + 0.5),
                &layout,
            );
            commands.spawn((
                Sprite {
                    color,
                    custom_size: Some(Vec2::splat(layout.tile - 1.0)),
                    ..default()
                },
                Transform::from_xyz(pos.x, pos.y, 0.0),
            ));
        }
    }

    for _ in 0..config.0.particle_count {
        commands.spawn((
            ParticleDot,
            Sprite {
                color: Color::srgba(0.4, 0.7, 1.0, 0.4),
                custom_size: Some(Vec2::splat(2.5)),
                ..default()
            },
            Transform::from_xyz(0.0, 0.0, 1.0),
        ));
    }

    for i in 0..config.0.n_rays {
        commands.spawn((
            LidarBeam(i),
            Sprite {
                color: Color::srgba(1.0, 0.55, 0.2, 0.35),
                custom_size: Some(Vec2::new(1.0, 1.5)),
                anchor: Anchor::CenterLeft,
                ..default()
            },
            Transform::from_xyz(0.0, 0.0, 2.0),
        ));
    }

    let triangle = meshes.add(Triangle2d::new(
        Vec2::new(0.6, 0.0),
        Vec2::new(-0.4, 0.4),
        Vec2::new(-0.4, -0.4),
    ));
    let arrow_size = layout.tile * 0.9;

    let estimate_mat = materials.add(Color::srgb(0.35, 1.0, 0.45));
    commands.spawn((
        Mesh2d(triangle.clone()),
        MeshMaterial2d(estimate_mat),
        Transform::from_xyz(0.0, 0.0, 3.0).with_scale(Vec3::splat(arrow_size)),
        EstimateMarker,
    ));

    let truth_mat = materials.add(Color::srgb(1.0, 0.88, 0.25));
    commands.spawn((
        Mesh2d(triangle),
        MeshMaterial2d(truth_mat),
        Transform::from_xyz(0.0, 0.0, 4.0).with_scale(Vec3::splat(arrow_size)),
        RobotMarker,
    ));
}

fn game_loop(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut robot: ResMut<AppRobot>,
    mut filter: ResMut<AppFilter>,
    mut rng: ResMut<AppRng>,
    mut last_scan: ResMut<AppLastScan>,
    map: Res<AppMap>,
    config: Res<AppConfig>,
) {
    let forward = if keyboard.pressed(KeyCode::ArrowUp) || keyboard.pressed(KeyCode::KeyW) {
        1.0_f32
    } else if keyboard.pressed(KeyCode::ArrowDown) || keyboard.pressed(KeyCode::KeyS) {
        -1.0_f32
    } else {
        0.0
    };

    let turn = if keyboard.pressed(KeyCode::ArrowLeft) || keyboard.pressed(KeyCode::KeyA) {
        -1.0_f32
    } else if keyboard.pressed(KeyCode::ArrowRight) || keyboard.pressed(KeyCode::KeyD) {
        1.0_f32
    } else {
        0.0
    };

    let speed = 0.05_f32;
    let theta = robot.0.theta;
    let delta = MotionDelta {
        dx: forward * speed * theta.cos(),
        dy: forward * speed * theta.sin(),
        dtheta: turn * 0.05,
    };

    robot.0.apply_motion(&delta, &map.0);
    let ranges = robot.0.get_scan(&map.0, config.0.n_rays);
    last_scan.0 = ranges.clone();
    filter.0.step(&delta, &ranges, &map.0, &mut rng.0);
}

fn sync_robot_marker(
    robot: Res<AppRobot>,
    layout: Res<Layout>,
    mut q: Query<&mut Transform, With<RobotMarker>>,
) {
    for mut t in q.iter_mut() {
        let pos = world_to_screen(Vec2::new(robot.0.x, robot.0.y), &layout);
        t.translation.x = pos.x;
        t.translation.y = pos.y;
        t.rotation = Quat::from_rotation_z(-robot.0.theta);
    }
}

fn sync_estimate_marker(
    filter: Res<AppFilter>,
    layout: Res<Layout>,
    mut q: Query<&mut Transform, (With<EstimateMarker>, Without<RobotMarker>)>,
) {
    let particles = filter.0.particles();
    if particles.is_empty() {
        return;
    }
    let n = particles.len() as f32;
    let mx = particles.iter().map(|p| p.x).sum::<f32>() / n;
    let my = particles.iter().map(|p| p.y).sum::<f32>() / n;
    let cs = particles.iter().map(|p| p.theta.cos()).sum::<f32>() / n;
    let sn = particles.iter().map(|p| p.theta.sin()).sum::<f32>() / n;
    let mtheta = sn.atan2(cs);

    for mut t in q.iter_mut() {
        let pos = world_to_screen(Vec2::new(mx, my), &layout);
        t.translation.x = pos.x;
        t.translation.y = pos.y;
        t.rotation = Quat::from_rotation_z(-mtheta);
    }
}

fn sync_lidar_beams(
    robot: Res<AppRobot>,
    last_scan: Res<AppLastScan>,
    layout: Res<Layout>,
    config: Res<AppConfig>,
    mut q: Query<(&LidarBeam, &mut Transform, &mut Sprite)>,
) {
    if last_scan.0.is_empty() {
        return;
    }
    let n = config.0.n_rays as f32;
    let origin = world_to_screen(Vec2::new(robot.0.x, robot.0.y), &layout);
    for (beam, mut t, mut sprite) in q.iter_mut() {
        let r = last_scan.0.get(beam.0).copied().unwrap_or(0.0);
        let world_angle = robot.0.theta + (beam.0 as f32 / n) * std::f32::consts::TAU;
        t.translation.x = origin.x;
        t.translation.y = origin.y;
        t.rotation = Quat::from_rotation_z(-world_angle);
        sprite.custom_size = Some(Vec2::new(r * layout.tile, 1.5));
    }
}

fn sync_particles(
    filter: Res<AppFilter>,
    layout: Res<Layout>,
    mut q: Query<(&mut Transform, &mut Sprite), With<ParticleDot>>,
) {
    let particles = filter.0.particles();
    if particles.is_empty() {
        return;
    }
    let n = particles.len() as f32;
    let cx = particles.iter().map(|p| p.x).sum::<f32>() / n;
    let cy = particles.iter().map(|p| p.y).sum::<f32>() / n;

    for ((mut t, mut sprite), p) in q.iter_mut().zip(particles.iter()) {
        let pos = world_to_screen(Vec2::new(p.x, p.y), &layout);
        t.translation.x = pos.x;
        t.translation.y = pos.y;
        let dx = p.x - cx;
        let dy = p.y - cy;
        let d = (dx * dx + dy * dy).sqrt();
        let closeness = (-d * 0.4).exp();
        sprite.color = Color::srgba(
            0.35 + 0.35 * closeness,
            0.7 + 0.3 * closeness,
            1.0,
            0.25 + 0.6 * closeness,
        );
    }
}

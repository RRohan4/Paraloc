use bevy::prelude::*;
use rand::{rngs::StdRng, SeedableRng};

use crate::{
    config::Config,
    filter::ParticleFilter,
    map::Map,
    particle::{MotionDelta, NoiseParams, Particle},
    robot::Robot,
    sensor_model::SequentialSensorModel,
};

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

#[derive(Component)]
struct ParticleDot;

pub struct RendererPlugin;

impl Plugin for RendererPlugin {
    fn build(&self, app: &mut App) {
        let config = Config::load_or_default("config.toml");
        let map = Map::from_png(&config.map_path);
        let mut rng = StdRng::seed_from_u64(0);

        let (w, h) = (map.width() as f32, map.height() as f32);
        let particles: Vec<Particle> = (0..config.particle_count).map(|_| {
            use rand::Rng;
            Particle {
                x: rng.gen_range(0.0..w),
                y: rng.gen_range(0.0..h),
                theta: rng.gen_range(0.0..std::f32::consts::TAU),
                weight: 1.0 / config.particle_count as f32,
            }
        }).collect();

        let model = Box::new(SequentialSensorModel {
            sigma: config.sensor_sigma,
            n_rays: config.n_rays,
        });
        let noise = NoiseParams {
            translation_sigma: config.translation_sigma,
            rotation_sigma: config.rotation_sigma,
        };
        let filter = ParticleFilter::new(particles, model, noise);
        let robot = Robot::new(w / 2.0, h / 2.0, 0.0);

        app.insert_resource(AppMap(map))
           .insert_resource(AppRobot(robot))
           .insert_resource(AppFilter(filter))
           .insert_resource(AppRng(rng))
           .insert_resource(AppConfig(config))
           .add_plugins(DefaultPlugins)
           .add_systems(Startup, setup)
           .add_systems(Update, (game_loop, sync_particles).chain());
    }
}

fn setup(
    mut commands: Commands,
    map: Res<AppMap>,
    config: Res<AppConfig>,
) {
    commands.spawn(Camera2d);

    let tile = 8.0_f32;

    for y in 0..map.0.height() {
        for x in 0..map.0.width() {
            let color = if map.0.is_wall(x as f32, y as f32) {
                Color::srgb(0.2, 0.2, 0.2)
            } else {
                Color::srgb(0.9, 0.9, 0.9)
            };
            commands.spawn((
                Sprite {
                    color,
                    custom_size: Some(Vec2::splat(tile - 1.0)),
                    ..default()
                },
                Transform::from_xyz(x as f32 * tile, y as f32 * tile, 0.0),
            ));
        }
    }

    for _ in 0..config.0.particle_count {
        commands.spawn((
            ParticleDot,
            Sprite {
                color: Color::srgb(0.0, 0.5, 1.0),
                custom_size: Some(Vec2::splat(3.0)),
                ..default()
            },
            Transform::from_xyz(0.0, 0.0, 1.0),
        ));
    }
}

fn game_loop(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut robot: ResMut<AppRobot>,
    mut filter: ResMut<AppFilter>,
    mut rng: ResMut<AppRng>,
    map: Res<AppMap>,
    config: Res<AppConfig>,
) {
    let forward = if keyboard.pressed(KeyCode::ArrowUp) { 1.0_f32 }
              else if keyboard.pressed(KeyCode::ArrowDown) { -1.0_f32 }
              else { 0.0 };

    let turn = if keyboard.pressed(KeyCode::ArrowLeft) { 1.0_f32 }
           else if keyboard.pressed(KeyCode::ArrowRight) { -1.0_f32 }
           else { 0.0 };

    let speed = 0.05_f32;
    let theta = robot.0.theta;
    let delta = MotionDelta {
        dx: forward * speed * theta.cos(),
        dy: forward * speed * theta.sin(),
        dtheta: turn * 0.05,
    };

    robot.0.apply_motion(&delta, &map.0);
    let ranges = robot.0.get_scan(&map.0, config.0.n_rays);
    filter.0.step(&delta, &ranges, &map.0, &mut rng.0);
}

fn sync_particles(
    filter: Res<AppFilter>,
    mut dots: Query<&mut Transform, With<ParticleDot>>,
) {
    let tile = 8.0_f32;
    for (mut transform, particle) in dots.iter_mut().zip(filter.0.particles().iter()) {
        transform.translation.x = particle.x * tile;
        transform.translation.y = particle.y * tile;
    }
}

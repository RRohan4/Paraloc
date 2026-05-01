use paraloc::{
    filter::ParticleFilter,
    map::Map,
    particle::{MotionDelta, NoiseParams, Particle},
    sensor_model::SequentialSensorModel,
};
use rand::{SeedableRng, rngs::StdRng};

fn walled_room() -> Map {
    let mut data = vec![0u8; 25];
    for x in 0..5 { data[x] = 1; data[20 + x] = 1; }
    for y in 0..5 { data[y * 5] = 1; data[y * 5 + 4] = 1; }
    Map::from_raw(data, 5, 5)
}

#[test]
fn filter_converges_toward_true_pose() {
    let mut rng = StdRng::seed_from_u64(42);
    let map = walled_room();

    let true_x = 2.5_f32;
    let true_y = 2.5_f32;
    let true_theta = 0.0_f32;
    let n_rays = 36;

    let real_ranges = paraloc::raycaster::scan((true_x, true_y), true_theta, n_rays, &map);

    let particles: Vec<Particle> = (0..500).map(|_| {
        use rand::Rng;
        Particle {
            x: rng.gen_range(1.1..3.9),
            y: rng.gen_range(1.1..3.9),
            theta: rng.gen_range(0.0..std::f32::consts::TAU),
            weight: 1.0 / 500.0,
        }
    }).collect();

    let model = Box::new(SequentialSensorModel { sigma: 0.2, n_rays });
    let noise = NoiseParams { translation_sigma: 0.0, rotation_sigma: 0.0 };
    let mut filter = ParticleFilter::new(particles, model, noise);

    let motion = MotionDelta { dx: 0.0, dy: 0.0, dtheta: 0.0 };
    for _ in 0..15 {
        filter.step(&motion, &real_ranges, &map, &mut rng);
    }

    let n = filter.particles().len() as f32;
    let mean_x = filter.particles().iter().map(|p| p.x).sum::<f32>() / n;
    let mean_y = filter.particles().iter().map(|p| p.y).sum::<f32>() / n;

    assert!((mean_x - true_x).abs() < 0.3, "mean_x={}", mean_x);
    assert!((mean_y - true_y).abs() < 0.3, "mean_y={}", mean_y);
}

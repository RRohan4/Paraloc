use rand::Rng;
use rand_distr::{Distribution, Normal};

#[derive(Clone, Debug)]
pub struct Particle {
    pub x: f32,
    pub y: f32,
    pub theta: f32,
    pub weight: f32,
}

pub struct MotionDelta {
    pub dx: f32,
    pub dy: f32,
    pub dtheta: f32,
}

pub struct NoiseParams {
    pub translation_sigma: f32,
    pub rotation_sigma: f32,
}

pub fn predict<R: Rng>(
    particles: &mut [Particle],
    motion: &MotionDelta,
    noise: &NoiseParams,
    rng: &mut R,
) {
    let noise_xy = Normal::new(0.0_f32, noise.translation_sigma).unwrap();
    let noise_th = Normal::new(0.0_f32, noise.rotation_sigma).unwrap();
    for p in particles.iter_mut() {
        p.x += motion.dx + noise_xy.sample(rng);
        p.y += motion.dy + noise_xy.sample(rng);
        p.theta += motion.dtheta + noise_th.sample(rng);
    }
}

pub fn resample<R: Rng>(particles: &[Particle], rng: &mut R) -> Vec<Particle> {
    let n = particles.len();
    let total: f32 = particles.iter().map(|p| p.weight).sum();
    let step = total / n as f32;
    let start: f32 = rng.gen::<f32>() * step;
    let mut result = Vec::with_capacity(n);
    let mut cumulative = 0.0_f32;
    let mut j = 0;
    for i in 0..n {
        let threshold = start + i as f32 * step;
        while cumulative + particles[j].weight < threshold {
            cumulative += particles[j].weight;
            j += 1;
        }
        let mut p = particles[j].clone();
        p.weight = 1.0 / n as f32;
        result.push(p);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    fn particle(x: f32, y: f32, theta: f32, weight: f32) -> Particle {
        Particle { x, y, theta, weight }
    }

    #[test]
    fn predict_with_zero_noise_applies_exact_delta() {
        let mut particles = vec![particle(1.0, 2.0, 0.5, 1.0)];
        let motion = MotionDelta { dx: 0.1, dy: 0.2, dtheta: 0.3 };
        let noise = NoiseParams { translation_sigma: 0.0, rotation_sigma: 0.0 };
        predict(&mut particles, &motion, &noise, &mut make_rng());
        assert!((particles[0].x - 1.1).abs() < 1e-5);
        assert!((particles[0].y - 2.2).abs() < 1e-5);
        assert!((particles[0].theta - 0.8).abs() < 1e-5);
    }

    #[test]
    fn predict_mean_matches_motion_delta_over_many_particles() {
        let n = 10_000;
        let mut particles = vec![particle(0.0, 0.0, 0.0, 1.0); n];
        let motion = MotionDelta { dx: 1.0, dy: -1.0, dtheta: 0.5 };
        let noise = NoiseParams { translation_sigma: 0.1, rotation_sigma: 0.1 };
        predict(&mut particles, &motion, &noise, &mut make_rng());
        let mean_x = particles.iter().map(|p| p.x).sum::<f32>() / n as f32;
        let mean_y = particles.iter().map(|p| p.y).sum::<f32>() / n as f32;
        assert!((mean_x - 1.0).abs() < 0.01, "mean_x={}", mean_x);
        assert!((mean_y - -1.0).abs() < 0.01, "mean_y={}", mean_y);
    }

    #[test]
    fn predict_stddev_matches_noise_params() {
        let n = 10_000;
        let sigma = 0.5_f32;
        let mut particles = vec![particle(0.0, 0.0, 0.0, 1.0); n];
        let motion = MotionDelta { dx: 0.0, dy: 0.0, dtheta: 0.0 };
        let noise = NoiseParams { translation_sigma: sigma, rotation_sigma: sigma };
        predict(&mut particles, &motion, &noise, &mut make_rng());
        let mean_x = particles.iter().map(|p| p.x).sum::<f32>() / n as f32;
        let var_x = particles.iter().map(|p| (p.x - mean_x).powi(2)).sum::<f32>() / n as f32;
        assert!((var_x.sqrt() - sigma).abs() < 0.02, "stddev={}", var_x.sqrt());
    }

    #[test]
    fn resample_concentrates_on_high_weight_particle() {
        let particles = vec![
            particle(0.0, 0.0, 0.0, 0.01),
            particle(9.0, 9.0, 0.0, 0.99),
        ];
        let result = resample(&particles, &mut make_rng());
        let high_weight_count = result.iter().filter(|p| p.x == 9.0).count();
        assert!(high_weight_count >= 1, "high-weight particle never selected");
    }

    #[test]
    fn resample_produces_uniform_weights() {
        let particles = vec![
            particle(0.0, 0.0, 0.0, 0.2),
            particle(1.0, 0.0, 0.0, 0.8),
        ];
        let result = resample(&particles, &mut make_rng());
        let expected = 1.0 / result.len() as f32;
        for p in &result {
            assert!((p.weight - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn resample_preserves_particle_count() {
        let particles = vec![
            particle(0.0, 0.0, 0.0, 0.25),
            particle(1.0, 0.0, 0.0, 0.25),
            particle(2.0, 0.0, 0.0, 0.25),
            particle(3.0, 0.0, 0.0, 0.25),
        ];
        let result = resample(&particles, &mut make_rng());
        assert_eq!(result.len(), particles.len());
    }
}

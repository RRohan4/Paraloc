use rand::Rng;

use crate::{
    map::Map,
    particle::{MotionDelta, NoiseParams, Particle, predict, resample},
    sensor_model::SensorModel,
};

const INJECTION_FRAC: f32 = 0.03;
const ESS_FRAC: f32 = 0.5;

pub struct ParticleFilter {
    particles: Vec<Particle>,
    weights: Vec<f32>,
    sensor_model: Box<dyn SensorModel>,
    noise_params: NoiseParams,
}

impl ParticleFilter {
    pub fn new(
        particles: Vec<Particle>,
        sensor_model: Box<dyn SensorModel>,
        noise_params: NoiseParams,
    ) -> Self {
        let n = particles.len();
        Self { particles, weights: vec![0.0_f32; n], sensor_model, noise_params }
    }

    pub fn step<R: Rng>(&mut self, motion: &MotionDelta, real_ranges: &[f32], map: &Map, rng: &mut R) {
        predict(&mut self.particles, motion, &self.noise_params, rng);

        self.sensor_model.update_weights(&self.particles, real_ranges, map, &mut self.weights);

        for (w, p) in self.weights.iter_mut().zip(self.particles.iter()) {
            *w += p.weight.max(f32::MIN_POSITIVE).ln();
        }

        let max_log = self.weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0_f32;
        for w in self.weights.iter_mut() {
            *w = (*w - max_log).exp();
            total += *w;
        }
        for (p, w) in self.particles.iter_mut().zip(self.weights.iter()) {
            p.weight = w / total;
        }

        let n = self.particles.len();
        let ess = 1.0 / self.particles.iter().map(|p| p.weight * p.weight).sum::<f32>();
        if ess < (n as f32) * ESS_FRAC {
            self.particles = resample(&self.particles, rng);
        }

        let n_inject = ((n as f32) * INJECTION_FRAC) as usize;
        if n_inject > 0 {
            let (mw_u, mh_u) = map.dimensions();
            let mw = mw_u as f32;
            let mh = mh_u as f32;
            let uniform_w = 1.0 / n as f32;
            for _ in 0..n_inject {
                let idx = rng.gen_range(0..n);
                let (x, y) = loop {
                    let cx = rng.gen_range(0.0..mw);
                    let cy = rng.gen_range(0.0..mh);
                    if !map.is_wall(cx, cy) {
                        break (cx, cy);
                    }
                };
                self.particles[idx] = Particle {
                    x,
                    y,
                    theta: rng.gen_range(0.0..std::f32::consts::TAU),
                    weight: uniform_w,
                };
            }
        }
    }

    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }
}

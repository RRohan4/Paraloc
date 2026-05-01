use rand::Rng;

use crate::{
    map::Map,
    particle::{MotionDelta, NoiseParams, Particle, predict, resample},
    sensor_model::SensorModel,
};

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

        let max_log = self.weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0_f32;
        for w in self.weights.iter_mut() {
            *w = (*w - max_log).exp();
            total += *w;
        }
        for (p, w) in self.particles.iter_mut().zip(self.weights.iter()) {
            p.weight = w / total;
        }

        self.particles = resample(&self.particles, rng);
    }

    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }
}

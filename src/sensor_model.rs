use crate::{map::Map, particle::Particle};

pub trait SensorModel: Send + Sync {
    fn update_weights(
        &self,
        particles: &[Particle],
        real_ranges: &[f32],
        map: &Map,
        weights: &mut [f32],
    );
}

pub struct SequentialSensorModel {
    pub sigma: f32,
    pub n_rays: usize,
}

impl SensorModel for SequentialSensorModel {
    fn update_weights(
        &self,
        particles: &[Particle],
        real_ranges: &[f32],
        map: &Map,
        weights: &mut [f32],
    ) {
        for (i, p) in particles.iter().enumerate() {
            let simulated = crate::raycaster::scan((p.x, p.y), p.theta, self.n_rays, map);
            let mut log_weight = 0.0_f32;
            for j in 0..self.n_rays {
                let diff = real_ranges[j] - simulated[j];
                log_weight += -0.5 * (diff / self.sigma).powi(2);
            }
            weights[i] = log_weight;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::Map;
    use crate::particle::Particle;

    fn walled_room() -> Map {
        let mut data = vec![0u8; 25];
        for x in 0..5 {
            data[x] = 1;
            data[20 + x] = 1;
        }
        for y in 0..5 {
            data[y * 5] = 1;
            data[y * 5 + 4] = 1;
        }
        Map::from_raw(data, 5, 5)
    }

    fn particle(x: f32, y: f32, theta: f32) -> Particle {
        Particle {
            x,
            y,
            theta,
            weight: 1.0,
        }
    }

    #[test]
    fn particle_at_true_pose_outweighs_distant_particle() {
        let map = walled_room();
        let real = crate::raycaster::scan((2.5, 2.5), 0.0, 36, &map);
        let model = SequentialSensorModel {
            sigma: 0.1,
            n_rays: 36,
        };
        let particles = vec![particle(2.5, 2.5, 0.0), particle(1.1, 1.1, 0.0)];
        let mut weights = vec![0.0_f32; 2];
        model.update_weights(&particles, &real, &map, &mut weights);
        assert!(weights[0] > weights[1], "true pose should score higher: {:?}", weights);
    }
}

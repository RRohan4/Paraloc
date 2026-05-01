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

pub struct ParallelSensorModel {
    pub sigma: f32,
    pub n_rays: usize,
}

impl SensorModel for ParallelSensorModel {
    fn update_weights(
        &self,
        particles: &[Particle],
        real_ranges: &[f32],
        map: &Map,
        weights: &mut [f32],
    ) {
        use rayon::prelude::*;
        particles.par_iter().zip(weights.par_iter_mut()).for_each(|(p, w)| {
            let simulated = crate::raycaster::scan((p.x, p.y), p.theta, self.n_rays, map);
            let mut log_weight = 0.0_f32;
            for j in 0..self.n_rays {
                let diff = real_ranges[j] - simulated[j];
                log_weight += -0.5 * (diff / self.sigma).powi(2);
            }
            *w = log_weight;
        });
    }
}

pub struct GpuSensorModel {
    pub sigma: f32,
    pub n_rays: usize,
}

impl SensorModel for GpuSensorModel {
    fn update_weights(
        &self,
        _particles: &[Particle],
        _real_ranges: &[f32],
        _map: &Map,
        _weights: &mut [f32],
    ) {
        todo!("GPU implementation")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::Map;
    use crate::particle::Particle;

    fn walled_room() -> Map {
        // 5x5 room, walls on edges, open interior
        let mut data = vec![0u8; 25];
        for x in 0..5 { data[x] = 1; data[20 + x] = 1; }
        for y in 0..5 { data[y * 5] = 1; data[y * 5 + 4] = 1; }
        Map::from_raw(data, 5, 5)
    }

    fn particle(x: f32, y: f32, theta: f32) -> Particle {
        Particle { x, y, theta, weight: 1.0 }
    }

    fn real_scan(map: &Map) -> Vec<f32> {
        crate::raycaster::scan((2.5, 2.5), 0.0, 36, map)
    }

    #[test]
    fn particle_at_true_pose_outweighs_distant_particle() {
        let map = walled_room();
        let real = real_scan(&map);
        let model = SequentialSensorModel { sigma: 0.1, n_rays: 36 };
        let particles = vec![particle(2.5, 2.5, 0.0), particle(1.1, 1.1, 0.0)];
        let mut weights = vec![0.0_f32; 2];
        model.update_weights(&particles, &real, &map, &mut weights);
        assert!(weights[0] > weights[1], "true pose should score higher: {:?}", weights);
    }

    #[test]
    fn identical_ranges_produce_maximum_weight() {
        let map = walled_room();
        let real = real_scan(&map);
        let model = SequentialSensorModel { sigma: 0.1, n_rays: 36 };
        let particles = vec![particle(2.5, 2.5, 0.0)];
        let mut weights = vec![0.0_f32; 1];
        model.update_weights(&particles, &real, &map, &mut weights);
        assert_eq!(weights[0], 0.0, "identical scan should give log-weight 0");
    }

    #[test]
    fn parallel_matches_sequential() {
        let map = walled_room();
        let real = real_scan(&map);
        let particles = vec![
            particle(2.5, 2.5, 0.0),
            particle(1.5, 1.5, 0.0),
            particle(3.0, 2.0, 1.0),
        ];
        let seq = SequentialSensorModel { sigma: 0.2, n_rays: 36 };
        let par = ParallelSensorModel { sigma: 0.2, n_rays: 36 };
        let mut w_seq = vec![0.0_f32; 3];
        let mut w_par = vec![0.0_f32; 3];
        seq.update_weights(&particles, &real, &map, &mut w_seq);
        par.update_weights(&particles, &real, &map, &mut w_par);
        for (s, p) in w_seq.iter().zip(w_par.iter()) {
            assert!((s - p).abs() < 1e-4, "seq={} par={}", s, p);
        }
    }
}

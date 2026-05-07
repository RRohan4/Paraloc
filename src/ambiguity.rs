use rand::Rng;

use crate::{
    config::Config,
    map::Map,
    particle::Particle,
    raycaster,
    sensor_model::{SequentialSensorModel, SensorModel},
};

const CROP_SIZE: usize = 64;

pub fn crop_window(map: &Map, center: (usize, usize), size: usize) -> Vec<u8> {
    let half = (size / 2) as isize;
    let mut crop = Vec::with_capacity(size * size);

    for row in 0..size as isize {
        for col in 0..size as isize {
            let src_x = center.0 as isize + col - half;
            let src_y = center.1 as isize + row - half;
            let is_wall = src_x < 0
                || src_y < 0
                || src_x >= map.width() as isize
                || src_y >= map.height() as isize
                || map.is_wall(src_x as f32 + 0.5, src_y as f32 + 0.5);
            crop.push(if is_wall { 255 } else { 0 });
        }
    }

    crop
}

pub fn default_crop_window(map: &Map, center: (usize, usize)) -> Vec<u8> {
    crop_window(map, center, CROP_SIZE)
}

pub fn open_cells(map: &Map) -> Vec<(usize, usize)> {
    let (width, height) = map.dimensions();
    let mut cells = Vec::new();

    for y in 0..height {
        for x in 0..width {
            if !map.is_wall(x as f32 + 0.5, y as f32 + 0.5) {
                cells.push((x, y));
            }
        }
    }

    cells
}

pub fn ambiguity_score<R: Rng>(query: (usize, usize), map: &Map, cfg: &Config, rng: &mut R) -> f32 {
    let nearby_cells = cells_within_radius(map, query, cfg.init_radius_tiles);
    if nearby_cells.is_empty() || cfg.n_score_particles <= 1 || cfg.n_score_orientations == 0 {
        return 0.0;
    }

    let query_origin = (query.0 as f32 + 0.5, query.1 as f32 + 0.5);
    let sensor_model = SequentialSensorModel {
        sigma: cfg.score_sensor_sigma,
        n_rays: cfg.n_rays,
    };

    let mut total_entropy = 0.0_f32;
    let orientation_count = cfg.n_score_orientations as f32;

    for orientation_idx in 0..cfg.n_score_orientations {
        let theta = orientation_idx as f32 * std::f32::consts::TAU / orientation_count;
        let scan = raycaster::scan(query_origin, theta, cfg.n_rays, map);
        let particles = sample_particles(&nearby_cells, theta, cfg.n_score_particles, rng);
        let mut log_weights = vec![0.0_f32; particles.len()];
        sensor_model.update_weights(&particles, &scan, map, &mut log_weights);
        total_entropy += normalized_entropy(&mut log_weights);
    }

    let max_entropy = (cfg.n_score_particles as f32).ln();
    if max_entropy <= 0.0 {
        0.0
    } else {
        (total_entropy / orientation_count / max_entropy).clamp(0.0, 1.0)
    }
}

fn cells_within_radius(map: &Map, center: (usize, usize), radius: usize) -> Vec<(usize, usize)> {
    let radius_i = radius as isize;
    let radius_sq = (radius * radius) as isize;
    let cx = center.0 as isize;
    let cy = center.1 as isize;
    let mut cells = Vec::new();

    for y in (cy - radius_i)..=(cy + radius_i) {
        for x in (cx - radius_i)..=(cx + radius_i) {
            if x < 0 || y < 0 || x >= map.width() as isize || y >= map.height() as isize {
                continue;
            }

            let dx = x - cx;
            let dy = y - cy;
            if dx * dx + dy * dy > radius_sq {
                continue;
            }

            if !map.is_wall(x as f32 + 0.5, y as f32 + 0.5) {
                cells.push((x as usize, y as usize));
            }
        }
    }

    cells
}

fn sample_particles<R: Rng>(
    candidates: &[(usize, usize)],
    theta: f32,
    count: usize,
    rng: &mut R,
) -> Vec<Particle> {
    let weight = 1.0 / count as f32;
    let mut particles = Vec::with_capacity(count);

    for _ in 0..count {
        let idx = rng.gen_range(0..candidates.len());
        let (x, y) = candidates[idx];
        particles.push(Particle {
            x: x as f32 + 0.5,
            y: y as f32 + 0.5,
            theta,
            weight,
        });
    }

    particles
}

fn normalized_entropy(log_weights: &mut [f32]) -> f32 {
    let max_log = log_weights
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0_f32;

    for weight in log_weights.iter_mut() {
        *weight = (*weight - max_log).exp();
        total += *weight;
    }

    if total <= 0.0 {
        return 0.0;
    }

    let mut entropy = 0.0_f32;
    for weight in log_weights.iter_mut() {
        *weight /= total;
        if *weight > 0.0 {
            entropy -= *weight * weight.ln();
        }
    }

    entropy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use rand::{rngs::StdRng, SeedableRng};

    fn simple_map() -> Map {
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

    #[test]
    fn crop_marks_walls_and_out_of_bounds() {
        let crop = crop_window(&simple_map(), (1, 1), 4);
        assert_eq!(crop.len(), 16);
        assert_eq!(crop[0], 255);
        assert_eq!(crop[10], 0);
    }

    #[test]
    fn ambiguity_score_is_normalized() {
        let map = simple_map();
        let mut rng = StdRng::seed_from_u64(7);
        let cfg = Config {
            n_score_particles: 16,
            n_score_orientations: 4,
            init_radius_tiles: 2,
            ..Config::default()
        };
        let score = ambiguity_score((2, 2), &map, &cfg, &mut rng);
        assert!((0.0..=1.0).contains(&score), "score={score}");
    }
}

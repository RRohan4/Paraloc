use crate::map::Map;

// Fire a single ray from origin at the given angle using DDA
pub fn cast_ray(origin: (f32, f32), angle: f32, map: &Map) -> f32 {
    let dx = angle.cos();
    let dy = angle.sin();
    let tile_x = origin.0 as i32;
    let tile_y = origin.1 as i32;
    let step_x = if dx == 0.0 { f32::INFINITY } else { 1.0 / dx.abs() };
    let step_y = if dy == 0.0 { f32::INFINITY } else { 1.0 / dy.abs() };
    let mut dist_x = if dx > 0.0 {
        (tile_x as f32 + 1.0 - origin.0) / dx.abs()
    } else if dx < 0.0 {
        (origin.0 - tile_x as f32) / dx.abs()
    } else {
        f32::INFINITY
    };
    let mut dist_y = if dy > 0.0 {
        (tile_y as f32 + 1.0 - origin.1) / dy.abs()
    } else if dy < 0.0 {
        (origin.1 - tile_y as f32) / dy.abs()
    } else {
        f32::INFINITY
    };
    let mut cur_x = tile_x;
    let mut cur_y = tile_y;
    loop {
        if dist_x < dist_y {
            dist_x += step_x;
            cur_x += if dx > 0.0 { 1 } else { -1 };
            if map.is_wall(cur_x as f32, cur_y as f32) {
                return dist_x - step_x;
            }
        } else {
            dist_y += step_y;
            cur_y += if dy > 0.0 { 1 } else { -1 };
            if map.is_wall(cur_x as f32, cur_y as f32) {
                return dist_y - step_y;
            }
        }
    }
}

// Fire n_rays evenly spread across 360 degrees starting from yaw
pub fn scan(origin: (f32, f32), yaw: f32, n_rays: usize, map: &Map) -> Vec<f32> {
    let mut dists: Vec<f32> = Vec::new();

    for i in 0..n_rays {
        let angle = yaw + (i as f32) * (2.0 * std::f32::consts::PI / n_rays as f32);
        dists.push(cast_ray(origin, angle, map));
    }

    dists
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::Map;

    fn open_room() -> Map {
        // 4x4: walls on all edges, open interior
        // W W W W
        // W . . W
        // W . . W
        // W W W W
        let mut data = vec![0u8; 16];
        for x in 0..4 {
            data[x] = 1;
            data[12 + x] = 1;
        }
        for y in 0..4 {
            data[y * 4] = 1;
            data[y * 4 + 3] = 1;
        }
        Map::from_raw(data, 4, 4)
    }

    #[test]
    fn ray_hits_wall_at_correct_distance() {
        // origin (1.5, 1.5) facing right (angle 0) — wall at x=3, distance 1.5
        let dist = cast_ray((1.5, 1.5), 0.0, &open_room());
        assert!((dist - 1.5).abs() < 1e-4, "expected 1.5, got {}", dist);
    }

    #[test]
    fn ray_facing_wall_directly_returns_near_zero() {
        // origin (1.01, 1.5) facing left — wall tile x=0, boundary at x=1.0, distance 0.01
        let dist = cast_ray((1.01, 1.5), std::f32::consts::PI, &open_room());
        assert!((dist - 0.01).abs() < 1e-3, "expected ~0.01, got {}", dist);
    }

    #[test]
    fn full_scan_in_symmetric_room_returns_equal_ranges() {
        // 5x5 walled room, robot centered at (2.5, 2.5) — all 4 axis rays equal
        let mut data = vec![0u8; 25];
        for x in 0..5 { data[x] = 1; data[20 + x] = 1; }
        for y in 0..5 { data[y * 5] = 1; data[y * 5 + 4] = 1; }
        let map = Map::from_raw(data, 5, 5);
        let ranges = scan((2.5, 2.5), 0.0, 4, &map);
        let first = ranges[0];
        for r in &ranges {
            assert!((r - first).abs() < 1e-3, "ranges not equal: {:?}", ranges);
        }
    }

    #[test]
    fn ray_never_passes_through_wall() {
        // no ray should travel further than the map diagonal
        let ranges = scan((1.5, 1.5), 0.0, 36, &open_room());
        for r in ranges {
            assert!(r < 6.0, "ray escaped map: {}", r);
        }
    }
}

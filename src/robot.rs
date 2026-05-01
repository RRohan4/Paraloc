use crate::{map::Map, particle::MotionDelta};

pub struct Robot {
    pub x: f32,
    pub y: f32,
    pub theta: f32,
}

impl Robot {
    pub fn new(x: f32, y: f32, theta: f32) -> Self {
        Robot {x, y, theta}
    }

    pub fn apply_motion(&mut self, delta: &MotionDelta, map: &Map) {
        if !map.is_wall(self.x + delta.dx, self.y + delta.dy) {    
            self.x += delta.dx;
            self.y += delta.dy;
        }
        self.theta += delta.dtheta;
    }

    pub fn get_scan(&self, map: &Map, n_rays: usize) -> Vec<f32> {
        crate::raycaster::scan((self.x, self.y), self.theta, n_rays, map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::Map;

    fn walled_room() -> Map {
        let mut data = vec![0u8; 25];
        for x in 0..5 { data[x] = 1; data[20 + x] = 1; }
        for y in 0..5 { data[y * 5] = 1; data[y * 5 + 4] = 1; }
        Map::from_raw(data, 5, 5)
    }

    #[test]
    fn motion_in_open_space_updates_pose() {
        let map = walled_room();
        let mut robot = Robot::new(2.5, 2.5, 0.0);
        let delta = MotionDelta { dx: 0.2, dy: 0.1, dtheta: 0.1 };
        robot.apply_motion(&delta, &map);
        assert!((robot.x - 2.7).abs() < 1e-5);
        assert!((robot.y - 2.6).abs() < 1e-5);
        assert!((robot.theta - 0.1).abs() < 1e-5);
    }

    #[test]
    fn motion_into_wall_is_blocked() {
        let map = walled_room();
        let mut robot = Robot::new(1.5, 2.5, 0.0);
        let delta = MotionDelta { dx: -1.0, dy: 0.0, dtheta: 0.0 };
        robot.apply_motion(&delta, &map);
        assert!((robot.x - 1.5).abs() < 1e-5, "x should not change: {}", robot.x);
        assert!((robot.y - 2.5).abs() < 1e-5);
    }

    #[test]
    fn scan_returns_correct_number_of_rays() {
        let map = walled_room();
        let robot = Robot::new(2.5, 2.5, 0.0);
        let scan = robot.get_scan(&map, 36);
        assert_eq!(scan.len(), 36);
    }
}

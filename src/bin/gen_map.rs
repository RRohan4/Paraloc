use std::collections::VecDeque;

use image::{GrayImage, Luma};
use rand::{rngs::StdRng, Rng, SeedableRng};

const WALL: Luma<u8> = Luma([0]);
const OPEN: Luma<u8> = Luma([255]);
const MAX_GENERATION_ATTEMPTS: usize = 96;

#[derive(Clone, Copy, Debug)]
struct Rect {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

impl Rect {
    fn left(self) -> u32 {
        self.x
    }

    fn top(self) -> u32 {
        self.y
    }

    fn right(self) -> u32 {
        self.x + self.w
    }

    fn bottom(self) -> u32 {
        self.y + self.h
    }

    fn center(self) -> (u32, u32) {
        (self.x + self.w / 2, self.y + self.h / 2)
    }

    fn area(self) -> u32 {
        self.w * self.h
    }

    fn inset(self, amount: u32) -> Option<Self> {
        if self.w <= amount * 2 || self.h <= amount * 2 {
            None
        } else {
            Some(Self {
                x: self.x + amount,
                y: self.y + amount,
                w: self.w - amount * 2,
                h: self.h - amount * 2,
            })
        }
    }

    fn intersects_with_margin(self, other: Self, margin: u32) -> bool {
        let left_a = self.left().saturating_sub(margin);
        let top_a = self.top().saturating_sub(margin);
        let right_a = (self.right() + margin).min(u32::MAX);
        let bottom_a = (self.bottom() + margin).min(u32::MAX);
        let left_b = other.left();
        let top_b = other.top();
        let right_b = other.right();
        let bottom_b = other.bottom();
        left_a < right_b && right_a > left_b && top_a < bottom_b && bottom_a > top_b
    }
}

#[derive(Clone, Debug)]
struct Grid {
    size: u32,
    cells: Vec<bool>,
}

impl Grid {
    fn new(size: u32) -> Self {
        Self {
            size,
            cells: vec![false; (size * size) as usize],
        }
    }

    fn index(&self, x: u32, y: u32) -> usize {
        (y * self.size + x) as usize
    }

    fn is_open(&self, x: u32, y: u32) -> bool {
        self.cells[self.index(x, y)]
    }

    fn set_open(&mut self, x: u32, y: u32) {
        if x > 0 && y > 0 && x + 1 < self.size && y + 1 < self.size {
            let idx = self.index(x, y);
            self.cells[idx] = true;
        }
    }

    fn set_wall(&mut self, x: u32, y: u32) {
        let idx = self.index(x, y);
        self.cells[idx] = false;
    }

    fn carve_rect(&mut self, rect: Rect) {
        for y in rect.top()..rect.bottom() {
            for x in rect.left()..rect.right() {
                self.set_open(x, y);
            }
        }
    }

    fn paint_wall_rect(&mut self, rect: Rect) {
        for y in rect.top()..rect.bottom() {
            for x in rect.left()..rect.right() {
                if x < self.size && y < self.size {
                    self.set_wall(x, y);
                }
            }
        }
    }

    fn carve_hallway_horizontal(&mut self, x0: u32, x1: u32, y: u32, width: u32) {
        let start = x0.min(x1);
        let end = x0.max(x1);
        let y0 = y.saturating_sub(width / 2).max(1);
        let y1 = (y0 + width).min(self.size - 1);
        self.carve_rect(Rect {
            x: start.max(1),
            y: y0,
            w: end.saturating_sub(start) + 1,
            h: y1.saturating_sub(y0),
        });
    }

    fn carve_hallway_vertical(&mut self, x: u32, y0: u32, y1: u32, width: u32) {
        let start = y0.min(y1);
        let end = y0.max(y1);
        let x0 = x.saturating_sub(width / 2).max(1);
        let x1 = (x0 + width).min(self.size - 1);
        self.carve_rect(Rect {
            x: x0,
            y: start.max(1),
            w: x1.saturating_sub(x0),
            h: end.saturating_sub(start) + 1,
        });
    }

    fn open_count(&self) -> usize {
        self.cells.iter().filter(|&&cell| cell).count()
    }

    fn perimeter_is_closed(&self) -> bool {
        let last = self.size - 1;
        (0..self.size).all(|x| !self.is_open(x, 0) && !self.is_open(x, last))
            && (0..self.size).all(|y| !self.is_open(0, y) && !self.is_open(last, y))
    }

    fn largest_component_size(&self) -> usize {
        let mut seen = vec![false; self.cells.len()];
        let mut best = 0_usize;
        let dirs = [(-1_i32, 0_i32), (1, 0), (0, -1), (0, 1)];

        for y in 1..self.size.saturating_sub(1) {
            for x in 1..self.size.saturating_sub(1) {
                if !self.is_open(x, y) {
                    continue;
                }
                let start_idx = self.index(x, y);
                if seen[start_idx] {
                    continue;
                }

                let mut queue = VecDeque::from([(x, y)]);
                seen[start_idx] = true;
                let mut count = 0_usize;
                while let Some((cx, cy)) = queue.pop_front() {
                    count += 1;
                    for (dx, dy) in dirs {
                        let nx = cx as i32 + dx;
                        let ny = cy as i32 + dy;
                        if nx <= 0
                            || ny <= 0
                            || nx >= self.size as i32 - 1
                            || ny >= self.size as i32 - 1
                        {
                            continue;
                        }
                        let nx = nx as u32;
                        let ny = ny as u32;
                        if !self.is_open(nx, ny) {
                            continue;
                        }
                        let idx = self.index(nx, ny);
                        if seen[idx] {
                            continue;
                        }
                        seen[idx] = true;
                        queue.push_back((nx, ny));
                    }
                }
                best = best.max(count);
            }
        }

        best
    }

    fn to_image(&self) -> GrayImage {
        let mut img = GrayImage::from_pixel(self.size, self.size, WALL);
        for y in 0..self.size {
            for x in 0..self.size {
                if self.is_open(x, y) {
                    img.put_pixel(x, y, OPEN);
                }
            }
        }
        img
    }
}

fn parse_args() -> (u64, u32, String) {
    let mut seed = 0_u64;
    let mut size = 128_u32;
    let mut output = "map.png".to_string();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seed" => {
                seed = args
                    .next()
                    .expect("missing value for --seed")
                    .parse()
                    .expect("invalid --seed value");
            }
            "--size" => {
                size = args
                    .next()
                    .expect("missing value for --size")
                    .parse()
                    .expect("invalid --size value");
            }
            "--output" => {
                output = args.next().expect("missing value for --output");
            }
            other => panic!("unknown arg: {other}"),
        }
    }

    assert!(size >= 64, "map size must be at least 64 tiles");
    (seed, size, output)
}

fn room_count_for_size(size: u32) -> usize {
    if size < 96 { 5 } else { 7 }
}

fn room_span_range(size: u32) -> (u32, u32) {
    let min = (size / 10).max(9);
    let max = (size / 5).max(min + 2);
    (min, max)
}

fn can_place_room(rooms: &[Rect], candidate: Rect, margin: u32, size: u32) -> bool {
    if candidate.left() <= 1
        || candidate.top() <= 1
        || candidate.right() >= size - 1
        || candidate.bottom() >= size - 1
    {
        return false;
    }
    rooms
        .iter()
        .all(|room| !room.intersects_with_margin(candidate, margin))
}

fn sample_room<R: Rng>(rng: &mut R, size: u32) -> Rect {
    let (min_span, max_span) = room_span_range(size);
    let w = rng.gen_range(min_span..=max_span);
    let h = rng.gen_range(min_span..=max_span);
    let x = rng.gen_range(2..=(size - w - 2));
    let y = rng.gen_range(2..=(size - h - 2));
    Rect { x, y, w, h }
}

fn central_anchor_room(size: u32) -> Rect {
    let w = (size / 5).max(12);
    let h = (size / 5).max(12);
    Rect {
        x: size / 2 - w / 2,
        y: size / 2 - h / 2,
        w,
        h,
    }
}

fn door_toward(room: Rect, target: (u32, u32), width: u32) -> Rect {
    let (cx, cy) = room.center();
    let dx = target.0 as i32 - cx as i32;
    let dy = target.1 as i32 - cy as i32;
    let half = width / 2;

    if dx.abs() >= dy.abs() {
        let y = target.1.clamp(room.top() + 2, room.bottom().saturating_sub(3));
        if dx >= 0 {
            Rect {
                x: room.right().saturating_sub(1),
                y: y.saturating_sub(half),
                w: 2,
                h: width.max(2),
            }
        } else {
            Rect {
                x: room.left(),
                y: y.saturating_sub(half),
                w: 2,
                h: width.max(2),
            }
        }
    } else {
        let x = target.0.clamp(room.left() + 2, room.right().saturating_sub(3));
        if dy >= 0 {
            Rect {
                x: x.saturating_sub(half),
                y: room.bottom().saturating_sub(1),
                w: width.max(2),
                h: 2,
            }
        } else {
            Rect {
                x: x.saturating_sub(half),
                y: room.top(),
                w: width.max(2),
                h: 2,
            }
        }
    }
}

fn connect_rooms<R: Rng>(grid: &mut Grid, from: Rect, to: Rect, width: u32, rng: &mut R) {
    let from_center = from.center();
    let to_center = to.center();
    let from_door = door_toward(from, to_center, width);
    let to_door = door_toward(to, from_center, width);
    grid.carve_rect(from_door);
    grid.carve_rect(to_door);

    let start = from_door.center();
    let end = to_door.center();
    if rng.gen_bool(0.5) {
        grid.carve_hallway_horizontal(start.0, end.0, start.1, width);
        grid.carve_hallway_vertical(end.0, start.1, end.1, width);
    } else {
        grid.carve_hallway_vertical(start.0, start.1, end.1, width);
        grid.carve_hallway_horizontal(start.0, end.0, end.1, width);
    }
}

fn add_room_clutter<R: Rng>(grid: &mut Grid, room: Rect, rng: &mut R) -> usize {
    let Some(interior) = room.inset(2) else {
        return 0;
    };
    if interior.area() < 90 {
        return 0;
    }

    let mut added = 0_usize;
    let feature_count = rng.gen_range(0..=2);
    for _ in 0..feature_count {
        if rng.gen_bool(0.55) {
            let pillar_w = rng.gen_range(2..=3).min(interior.w.saturating_sub(1));
            let pillar_h = rng.gen_range(2..=3).min(interior.h.saturating_sub(1));
            if pillar_w < 2 || pillar_h < 2 {
                continue;
            }
            let x = rng.gen_range(interior.left()..=(interior.right() - pillar_w));
            let y = rng.gen_range(interior.top()..=(interior.bottom() - pillar_h));
            grid.paint_wall_rect(Rect {
                x,
                y,
                w: pillar_w,
                h: pillar_h,
            });
            added += 1;
        } else {
            let horizontal = rng.gen_bool(0.5);
            if horizontal && interior.w > 6 {
                let length = rng.gen_range((interior.w / 3).max(3)..=(interior.w / 2).max(4));
                let x = rng.gen_range(interior.left()..=(interior.right() - length));
                let y = rng.gen_range(interior.top() + 1..=(interior.bottom() - 2));
                let door_x = x + length / 2;
                grid.paint_wall_rect(Rect {
                    x,
                    y,
                    w: length,
                    h: 1,
                });
                grid.carve_rect(Rect {
                    x: door_x.saturating_sub(1),
                    y,
                    w: 2,
                    h: 1,
                });
                added += 1;
            } else if interior.h > 6 {
                let length = rng.gen_range((interior.h / 3).max(3)..=(interior.h / 2).max(4));
                let x = rng.gen_range(interior.left() + 1..=(interior.right() - 2));
                let y = rng.gen_range(interior.top()..=(interior.bottom() - length));
                let door_y = y + length / 2;
                grid.paint_wall_rect(Rect {
                    x,
                    y,
                    w: 1,
                    h: length,
                });
                grid.carve_rect(Rect {
                    x,
                    y: door_y.saturating_sub(1),
                    w: 1,
                    h: 2,
                });
                added += 1;
            }
        }
    }

    added
}

fn build_candidate<R: Rng>(rng: &mut R, size: u32) -> Option<Grid> {
    let mut grid = Grid::new(size);
    let mut rooms = Vec::new();
    let anchor = central_anchor_room(size);
    rooms.push(anchor);
    grid.carve_rect(anchor);

    let target_rooms = room_count_for_size(size);
    let placement_margin = (size / 32).max(3);
    let mut attempts = 0_usize;
    while rooms.len() < target_rooms && attempts < 256 {
        attempts += 1;
        let room = sample_room(rng, size);
        if can_place_room(&rooms, room, placement_margin, size) {
            grid.carve_rect(room);
            rooms.push(room);
        }
    }

    if rooms.len() < 4 {
        return None;
    }

    let hall_width = rng.gen_range(3..=5);
    for idx in 1..rooms.len() {
        let room = rooms[idx];
        let (cx, cy) = room.center();
        let connect_to = (0..idx)
            .min_by_key(|&other_idx| {
                let (ox, oy) = rooms[other_idx].center();
                (cx as i32 - ox as i32).unsigned_abs() + (cy as i32 - oy as i32).unsigned_abs()
            })
            .unwrap();
        connect_rooms(&mut grid, room, rooms[connect_to], hall_width, rng);
    }

    let extra_links = rng.gen_range(1..=2).min(rooms.len().saturating_sub(1));
    for _ in 0..extra_links {
        let a = rng.gen_range(0..rooms.len());
        let mut b = rng.gen_range(0..rooms.len());
        while b == a {
            b = rng.gen_range(0..rooms.len());
        }
        connect_rooms(&mut grid, rooms[a], rooms[b], hall_width, rng);
    }

    let mut clutter_features = 0_usize;
    for room in &rooms {
        clutter_features += add_room_clutter(&mut grid, *room, rng);
    }

    if clutter_features == 0 && !rooms.is_empty() {
        let _ = add_room_clutter(&mut grid, rooms[0], rng);
    }

    if is_valid_layout(&grid, rooms.len()) {
        Some(grid)
    } else {
        None
    }
}

fn is_valid_layout(grid: &Grid, room_count: usize) -> bool {
    if !grid.perimeter_is_closed() {
        return false;
    }
    if room_count < 4 {
        return false;
    }

    let open_count = grid.open_count();
    if open_count == 0 {
        return false;
    }

    let total = (grid.size * grid.size) as usize;
    let open_ratio = open_count as f32 / total as f32;
    if !(0.20..=0.58).contains(&open_ratio) {
        return false;
    }

    let largest_component = grid.largest_component_size();
    if largest_component * 100 < open_count * 92 {
        return false;
    }

    largest_component > (grid.size as usize * 6)
}

fn generate_map(seed: u64, size: u32) -> Grid {
    let mut rng = StdRng::seed_from_u64(seed);
    for attempt in 0..MAX_GENERATION_ATTEMPTS {
        let attempt_seed = rng.gen::<u64>()
            ^ (attempt as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut attempt_rng = StdRng::seed_from_u64(attempt_seed);
        if let Some(grid) = build_candidate(&mut attempt_rng, size) {
            return grid;
        }
    }

    panic!("failed to generate a valid map after {MAX_GENERATION_ATTEMPTS} attempts");
}

fn main() {
    let (seed, size, output) = parse_args();
    let img = generate_map(seed, size).to_image();
    img.save(&output).unwrap();
    println!("wrote {output} ({size}x{size}, seed={seed})");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boundary_is_closed(grid: &Grid) -> bool {
        grid.perimeter_is_closed()
    }

    #[test]
    fn generated_maps_keep_closed_boundaries() {
        for seed in [1_u64, 7, 42, 101, 999] {
            let grid = generate_map(seed, 128);
            assert!(boundary_is_closed(&grid), "seed {seed} leaked through boundary");
            assert!(is_valid_layout(&grid, room_count_for_size(128)));
        }
    }

    #[test]
    fn different_seeds_produce_distinct_layouts() {
        let first = generate_map(11, 128).cells;
        let second = generate_map(12, 128).cells;
        let third = generate_map(13, 128).cells;
        assert_ne!(first, second);
        assert_ne!(second, third);
        assert_ne!(first, third);
    }

    #[test]
    fn generated_maps_have_meaningful_open_space() {
        for seed in [3_u64, 8, 21, 34] {
            let grid = generate_map(seed, 128);
            let open = grid.open_count();
            let total = (grid.size * grid.size) as usize;
            let ratio = open as f32 / total as f32;
            assert!(ratio > 0.20 && ratio < 0.58, "seed {seed} ratio {ratio}");
            assert!(grid.largest_component_size() * 100 >= open * 92);
        }
    }
}

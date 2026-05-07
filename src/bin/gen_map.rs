use image::{GrayImage, Luma};
use rand::{rngs::StdRng, Rng, SeedableRng};

const WALL: Luma<u8> = Luma([0]);
const OPEN: Luma<u8> = Luma([255]);

fn fill(img: &mut GrayImage, x0: u32, y0: u32, x1: u32, y1: u32, value: Luma<u8>) {
    for y in y0..y1 {
        for x in x0..x1 {
            img.put_pixel(x, y, value);
        }
    }
}

fn clamp_range(start: i32, end: i32, size: u32) -> (u32, u32) {
    let min = 1_i32;
    let max = size as i32 - 1;
    let x0 = start.clamp(min, max) as u32;
    let x1 = end.clamp(min, max) as u32;
    (x0.min(x1), x0.max(x1))
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

fn main() {
    let (seed, size, output) = parse_args();
    let mut img = GrayImage::from_pixel(size, size, WALL);
    let mut rng = StdRng::seed_from_u64(seed);

    let corridor_height = ((size as f32) * 0.46) as u32;
    let connector_top = corridor_height.saturating_sub(3);
    let hall_top = corridor_height + 2;
    fill(&mut img, 1, hall_top, size - 1, size - 1, OPEN);

    let base_centers = [
        (size as f32 * 0.16) as i32,
        (size as f32 * 0.50) as i32,
        (size as f32 * 0.84) as i32,
    ];
    let base_width = (size / 10).max(8) as i32;
    let jitter = ((size as f32) * 0.02) as i32;

    let mut corridors = Vec::new();
    for center in base_centers {
        let width = base_width + rng.gen_range(-2..=2);
        let shift = rng.gen_range(-jitter..=jitter);
        let start = center + shift - width / 2;
        let end = center + shift + width / 2;
        let (x0, x1) = clamp_range(start, end, size);
        corridors.push((x0, x1.max(x0 + 1)));
    }

    for &(x0, x1) in &corridors {
        fill(&mut img, x0, 1, x1, corridor_height, OPEN);
        fill(&mut img, x0, connector_top, x1, hall_top + 1, OPEN);
    }

    for &(x0, x1) in &corridors {
        let right_notch = x1.saturating_sub(1);
        let left_notch = x0;
        let upper_notch_y = ((corridor_height as f32) * 0.32) as u32;
        let lower_notch_y = ((corridor_height as f32) * 0.67) as u32;
        let notch_h = (size / 20).max(3);
        for y in upper_notch_y..(upper_notch_y + notch_h).min(corridor_height) {
            img.put_pixel(right_notch, y, WALL);
        }
        for y in lower_notch_y..(lower_notch_y + notch_h).min(corridor_height) {
            img.put_pixel(left_notch, y, WALL);
        }
    }

    let diag_len = size / 3;
    let diag_start_x = (size as f32 * 0.22) as u32;
    let diag_start_y = (size as f32 * 0.58) as u32;
    for i in 0..diag_len {
        let x = diag_start_x + i;
        let y = diag_start_y + i;
        if x + 1 < size - 1 && y < size - 1 {
            img.put_pixel(x, y, WALL);
            img.put_pixel(x + 1, y, WALL);
        }
    }

    let l_wall_x0 = (size as f32 * 0.60) as u32;
    let l_wall_x1 = (size as f32 * 0.84) as u32;
    let l_wall_y = (size as f32 * 0.70) as u32;
    for x in l_wall_x0..l_wall_x1 {
        img.put_pixel(x, l_wall_y, WALL);
    }
    for y in l_wall_y..(size as f32 * 0.90) as u32 {
        img.put_pixel(l_wall_x1.saturating_sub(1), y, WALL);
    }

    fill(
        &mut img,
        1,
        (size as f32 * 0.76) as u32,
        (size as f32 * 0.07) as u32,
        (size as f32 * 0.90) as u32,
        WALL,
    );
    img.put_pixel(
        (size as f32 * 0.07) as u32,
        (size as f32 * 0.82) as u32,
        WALL,
    );

    fill(
        &mut img,
        (size as f32 * 0.16) as u32,
        (size as f32 * 0.60) as u32,
        (size as f32 * 0.22) as u32,
        (size as f32 * 0.66) as u32,
        WALL,
    );

    fill(
        &mut img,
        (size as f32 * 0.88) as u32,
        (size as f32 * 0.60) as u32,
        (size as f32 * 0.94) as u32,
        (size as f32 * 0.64) as u32,
        WALL,
    );
    img.put_pixel(
        (size as f32 * 0.93) as u32,
        (size as f32 * 0.64) as u32,
        WALL,
    );

    for x in 0..size {
        img.put_pixel(x, 0, WALL);
        img.put_pixel(x, size - 1, WALL);
    }
    for y in 0..size {
        img.put_pixel(0, y, WALL);
        img.put_pixel(size - 1, y, WALL);
    }

    img.save(&output).unwrap();
    println!("wrote {output} ({size}x{size}, seed={seed})");
}

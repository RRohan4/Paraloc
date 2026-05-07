use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use image::{GrayImage, Rgb, RgbImage};
use paraloc::{
    ambiguity::{ambiguity_score, default_crop_window, open_cells},
    config::Config,
    map::Map,
};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;

const CROP_SIZE: usize = 64;
const PREVIEW_GRID_COLS: usize = 4;
const PREVIEW_GRID_ROWS: usize = 4;
const PREVIEW_TILE_SCALE: u32 = 4;
const SMOOTH_HEATMAP_SCALE: u32 = 12;
const SMOOTH_SIGMA_TILES: f32 = 0.9;

#[derive(Clone)]
struct SampleRecord {
    cell: (usize, usize),
    score: f32,
    crop: Vec<u8>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (map_path, override_output_dir) = parse_args();
    let mut cfg = Config::load_or_default("config.toml");
    if let Some(path) = override_output_dir {
        cfg.output_dir = path;
    }
    if let Some(path) = map_path {
        cfg.map_path = path;
    }

    let map = Map::from_path(&cfg.map_path);
    let (width, height) = map.dimensions();
    if width != cfg.map_size || height != cfg.map_size {
        return Err(format!(
            "expected {}x{} map per guide, got {}x{} for {}",
            cfg.map_size, cfg.map_size, width, height, cfg.map_path
        )
        .into());
    }

    let out_dir = dataset_dir(&cfg.output_dir, &cfg.map_path)?;
    fs::create_dir_all(&out_dir)?;

    let open = open_cells(&map);
    let samples: Vec<SampleRecord> = open
        .par_iter()
        .map(|&(x, y)| {
            let mut rng = StdRng::seed_from_u64(((y as u64) << 32) ^ (x as u64) ^ 0x5EED5EED);
            SampleRecord {
                cell: (x, y),
                score: ambiguity_score((x, y), &map, &cfg, &mut rng),
                crop: default_crop_window(&map, (x, y)),
            }
        })
        .collect();

    write_binary_outputs(&out_dir, &samples)?;
    write_map_preview(&out_dir, &map)?;
    write_score_heatmap(&out_dir, &map, &samples)?;
    write_smoothed_score_heatmap(&out_dir, &map, &samples)?;
    write_crop_preview(&out_dir, &samples)?;
    write_summary(&out_dir, &cfg.map_path, &samples, (width, height))?;

    println!("wrote dataset preview to {}", out_dir.display());
    println!(
        "files: crops.bin, scores.bin, map_preview.png, score_heatmap.png, score_heatmap_smooth.png, crop_preview.png, crop_preview.txt, summary.txt"
    );
    Ok(())
}

fn parse_args() -> (Option<String>, Option<String>) {
    let mut map_path = None;
    let mut output_dir = None;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--map" => map_path = Some(args.next().expect("missing value for --map")),
            "--output-dir" => {
                output_dir = Some(args.next().expect("missing value for --output-dir"))
            }
            other => panic!("unknown arg: {other}"),
        }
    }

    (map_path, output_dir)
}

fn dataset_dir(base: &str, map_path: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let stem = Path::new(map_path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or("invalid map file name")?;
    Ok(Path::new(base).join(stem))
}

fn write_binary_outputs(
    out_dir: &Path,
    samples: &[SampleRecord],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut crop_writer = BufWriter::new(File::create(out_dir.join("crops.bin"))?);
    let mut score_writer = BufWriter::new(File::create(out_dir.join("scores.bin"))?);

    for sample in samples {
        crop_writer.write_all(&sample.crop)?;
        score_writer.write_all(&sample.score.to_le_bytes())?;
    }

    crop_writer.flush()?;
    score_writer.flush()?;
    Ok(())
}

fn write_map_preview(out_dir: &Path, map: &Map) -> Result<(), Box<dyn std::error::Error>> {
    let (width, height) = map.dimensions();
    let mut img = GrayImage::new(
        width as u32 * PREVIEW_TILE_SCALE,
        height as u32 * PREVIEW_TILE_SCALE,
    );

    for y in 0..height {
        for x in 0..width {
            let pixel = if map.is_wall(x as f32 + 0.5, y as f32 + 0.5) {
                0
            } else {
                255
            };
            paint_block(
                &mut img,
                x as u32 * PREVIEW_TILE_SCALE,
                y as u32 * PREVIEW_TILE_SCALE,
                PREVIEW_TILE_SCALE,
                [pixel],
            );
        }
    }

    img.save(out_dir.join("map_preview.png"))?;
    Ok(())
}

fn write_score_heatmap(
    out_dir: &Path,
    map: &Map,
    samples: &[SampleRecord],
) -> Result<(), Box<dyn std::error::Error>> {
    let (scores, min_score, max_score) = build_score_grid(map, samples);
    let (width, height) = map.dimensions();
    let mut heatmap = RgbImage::new(
        width as u32 * PREVIEW_TILE_SCALE,
        height as u32 * PREVIEW_TILE_SCALE,
    );
    let score_range = (max_score - min_score).max(f32::EPSILON);

    for y in 0..height {
        for x in 0..width {
            let color = if map.is_wall(x as f32 + 0.5, y as f32 + 0.5) {
                [0, 0, 0]
            } else {
                let score = scores[y * width + x].unwrap_or(max_score);
                let ambiguity = ((score - min_score) / score_range).clamp(0.0, 1.0);
                color_ramp(ambiguity)
            };
            paint_block_rgb(
                &mut heatmap,
                x as u32 * PREVIEW_TILE_SCALE,
                y as u32 * PREVIEW_TILE_SCALE,
                PREVIEW_TILE_SCALE,
                color,
            );
        }
    }

    heatmap.save(out_dir.join("score_heatmap.png"))?;
    Ok(())
}

fn write_smoothed_score_heatmap(
    out_dir: &Path,
    map: &Map,
    samples: &[SampleRecord],
) -> Result<(), Box<dyn std::error::Error>> {
    let (scores, min_score, max_score) = build_score_grid(map, samples);
    let smoothed_scores = smooth_score_grid(map, &scores);
    let (width, height) = map.dimensions();
    let out_w = width as u32 * SMOOTH_HEATMAP_SCALE;
    let out_h = height as u32 * SMOOTH_HEATMAP_SCALE;
    let mut heatmap = RgbImage::new(out_w, out_h);
    let score_range = (max_score - min_score).max(f32::EPSILON);

    for py in 0..out_h {
        for px in 0..out_w {
            let cell_x = (px / SMOOTH_HEATMAP_SCALE).min(width as u32 - 1) as usize;
            let cell_y = (py / SMOOTH_HEATMAP_SCALE).min(height as u32 - 1) as usize;
            let color = if map.is_wall(cell_x as f32 + 0.5, cell_y as f32 + 0.5) {
                [0, 0, 0]
            } else {
                let src_x = (px as f32 + 0.5) / SMOOTH_HEATMAP_SCALE as f32 - 0.5;
                let src_y = (py as f32 + 0.5) / SMOOTH_HEATMAP_SCALE as f32 - 0.5;
                let score = sample_open_score(map, &smoothed_scores, cell_x, cell_y, src_x, src_y)
                    .unwrap_or(max_score);
                let ambiguity = ((score - min_score) / score_range).clamp(0.0, 1.0);
                color_ramp(ambiguity)
            };
            heatmap.put_pixel(px, py, Rgb(color));
        }
    }

    heatmap.save(out_dir.join("score_heatmap_smooth.png"))?;
    Ok(())
}

fn build_score_grid(map: &Map, samples: &[SampleRecord]) -> (Vec<Option<f32>>, f32, f32) {
    let (width, height) = map.dimensions();
    let mut scores = vec![None; width * height];
    let mut min_score = f32::INFINITY;
    let mut max_score = f32::NEG_INFINITY;

    for sample in samples {
        scores[sample.cell.1 * width + sample.cell.0] = Some(sample.score);
        min_score = min_score.min(sample.score);
        max_score = max_score.max(sample.score);
    }

    (scores, min_score, max_score)
}

fn smooth_score_grid(map: &Map, scores: &[Option<f32>]) -> Vec<Option<f32>> {
    let (width, height) = map.dimensions();
    let kernel_radius = (SMOOTH_SIGMA_TILES * 3.0).ceil() as isize;
    let mut smoothed = vec![None; scores.len()];

    for y in 0..height {
        for x in 0..width {
            if map.is_wall(x as f32 + 0.5, y as f32 + 0.5) {
                continue;
            }

            let mut weighted = 0.0_f32;
            let mut total_weight = 0.0_f32;
            for ny in y as isize - kernel_radius..=y as isize + kernel_radius {
                for nx in x as isize - kernel_radius..=x as isize + kernel_radius {
                    if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                        continue;
                    }
                    let ux = nx as usize;
                    let uy = ny as usize;
                    if map.is_wall(ux as f32 + 0.5, uy as f32 + 0.5) {
                        continue;
                    }
                    if let Some(score) = scores[uy * width + ux] {
                        let dx = x as f32 - ux as f32;
                        let dy = y as f32 - uy as f32;
                        let dist_sq = dx * dx + dy * dy;
                        let weight =
                            (-dist_sq / (2.0 * SMOOTH_SIGMA_TILES * SMOOTH_SIGMA_TILES)).exp();
                        weighted += score * weight;
                        total_weight += weight;
                    }
                }
            }

            if total_weight > 0.0 {
                smoothed[y * width + x] = Some(weighted / total_weight);
            }
        }
    }

    smoothed
}

fn sample_open_score(
    map: &Map,
    scores: &[Option<f32>],
    fallback_x: usize,
    fallback_y: usize,
    src_x: f32,
    src_y: f32,
) -> Option<f32> {
    let x0 = src_x.floor() as isize;
    let y0 = src_y.floor() as isize;
    let mut weighted = 0.0_f32;
    let mut total_weight = 0.0_f32;

    let neighbors = [(x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1)];

    for &(nx, ny) in &neighbors {
        if nx < 0 || ny < 0 || nx >= map.width() as isize || ny >= map.height() as isize {
            continue;
        }
        let ux = nx as usize;
        let uy = ny as usize;
        if map.is_wall(ux as f32 + 0.5, uy as f32 + 0.5) {
            continue;
        }
        if let Some(score) = scores[uy * map.width() + ux] {
            let dx = src_x - nx as f32;
            let dy = src_y - ny as f32;
            let dist_sq = dx * dx + dy * dy;
            let weight = 1.0 / (dist_sq + 1e-4);
            weighted += score * weight;
            total_weight += weight;
        }
    }

    if total_weight > 0.0 {
        Some(weighted / total_weight)
    } else {
        scores[fallback_y * map.width() + fallback_x]
    }
}

fn color_ramp(ambiguity: f32) -> [u8; 3] {
    let dark_red = [96.0, 0.0, 0.0];
    let white = [255.0, 255.0, 255.0];
    let t = ambiguity.clamp(0.0, 1.0);

    [
        lerp(white[0], dark_red[0], t) as u8,
        lerp(white[1], dark_red[1], t) as u8,
        lerp(white[2], dark_red[2], t) as u8,
    ]
}

fn lerp(start: f32, end: f32, t: f32) -> f32 {
    start + (end - start) * t
}

fn write_crop_preview(
    out_dir: &Path,
    samples: &[SampleRecord],
) -> Result<(), Box<dyn std::error::Error>> {
    let preview_count = (PREVIEW_GRID_COLS * PREVIEW_GRID_ROWS).min(samples.len());
    let mut preview = GrayImage::new(
        (PREVIEW_GRID_COLS * CROP_SIZE) as u32,
        (PREVIEW_GRID_ROWS * CROP_SIZE) as u32,
    );
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

    let step = if preview_count <= 1 {
        0.0
    } else {
        (sorted.len() - 1) as f32 / (preview_count - 1) as f32
    };
    let mut manifest = String::new();

    for slot in 0..preview_count {
        let sample_idx = (slot as f32 * step).round() as usize;
        let sample = &sorted[sample_idx];
        let grid_x = slot % PREVIEW_GRID_COLS;
        let grid_y = slot / PREVIEW_GRID_COLS;
        let base_x = grid_x * CROP_SIZE;
        let base_y = grid_y * CROP_SIZE;

        for row in 0..CROP_SIZE {
            for col in 0..CROP_SIZE {
                let value = sample.crop[row * CROP_SIZE + col];
                preview.put_pixel(
                    (base_x + col) as u32,
                    (base_y + row) as u32,
                    image::Luma([value]),
                );
            }
        }

        manifest.push_str(&format!(
            "#{slot}: cell=({}, {}), score={:.4}\n",
            sample.cell.0, sample.cell.1, sample.score
        ));
    }

    preview.save(out_dir.join("crop_preview.png"))?;
    fs::write(out_dir.join("crop_preview.txt"), manifest)?;
    Ok(())
}

fn write_summary(
    out_dir: &Path,
    map_path: &str,
    samples: &[SampleRecord],
    map_size: (usize, usize),
) -> Result<(), Box<dyn std::error::Error>> {
    let mut min_score = f32::INFINITY;
    let mut max_score = f32::NEG_INFINITY;
    let mut total_score = 0.0_f32;

    for sample in samples {
        min_score = min_score.min(sample.score);
        max_score = max_score.max(sample.score);
        total_score += sample.score;
    }

    let mean_score = if samples.is_empty() {
        0.0
    } else {
        total_score / samples.len() as f32
    };

    let summary = format!(
        "map={map_path}\nmap_size={}x{}\nsample_count={}\ncrop_size={}x{}\nmin_score={:.4}\nmean_score={:.4}\nmax_score={:.4}\n",
        map_size.0,
        map_size.1,
        samples.len(),
        CROP_SIZE,
        CROP_SIZE,
        min_score,
        mean_score,
        max_score
    );
    fs::write(out_dir.join("summary.txt"), summary)?;
    Ok(())
}

fn paint_block(img: &mut GrayImage, start_x: u32, start_y: u32, size: u32, value: [u8; 1]) {
    for y in start_y..start_y + size {
        for x in start_x..start_x + size {
            img.put_pixel(x, y, image::Luma(value));
        }
    }
}

fn paint_block_rgb(img: &mut RgbImage, start_x: u32, start_y: u32, size: u32, value: [u8; 3]) {
    for y in start_y..start_y + size {
        for x in start_x..start_x + size {
            img.put_pixel(x, y, Rgb(value));
        }
    }
}

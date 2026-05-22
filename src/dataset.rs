use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use image::{GrayImage, Rgb, RgbImage};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use serde::Serialize;

use crate::{
    ambiguity::{ambiguity_score, default_crop_window, open_cells},
    config::Config,
    map::Map,
};

pub const CROP_SIZE: usize = 64;
pub const WALL_SCORE_SENTINEL: f32 = -1.0;

const PREVIEW_GRID_COLS: usize = 4;
const PREVIEW_GRID_ROWS: usize = 4;
const PREVIEW_TILE_SCALE: u32 = 4;
const SMOOTH_HEATMAP_SCALE: u32 = 12;
const SMOOTH_SIGMA_TILES: f32 = 0.9;

#[derive(Clone)]
pub struct DatasetSample {
    pub cell: (usize, usize),
    pub score: f32,
    pub crop: Vec<u8>,
}

#[derive(Clone, Debug, Serialize)]
pub struct MapDatasetEntry {
    pub map_id: String,
    pub map_path: String,
    pub rel_dir: String,
    pub sample_count: usize,
    pub crop_offset: u64,
    pub score_offset: u64,
    pub score_grid_path: String,
    pub min_score: f32,
    pub mean_score: f32,
    pub max_score: f32,
}

#[derive(Debug, Serialize)]
pub struct DatasetIndex {
    pub version: u32,
    pub crop_size: usize,
    pub map_size: usize,
    pub crop_bytes: usize,
    pub score_dtype: &'static str,
    pub crop_encoding: &'static str,
    pub wall_score_sentinel: f32,
    pub n_maps: usize,
    pub n_samples: usize,
    pub crops_file: String,
    pub scores_file: String,
    pub manifest_file: String,
    pub maps: Vec<MapDatasetEntry>,
}

pub struct GenerateOptions {
    pub write_previews: bool,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            write_previews: true,
        }
    }
}

pub fn output_dir_for_map(base: &Path, map_path: &Path) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    let stem = map_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or("invalid map file name")?;
    Ok(base.join(stem))
}

pub fn collect_map_pngs(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error + Send + Sync>> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "png"))
        .collect();
    paths.sort();
    Ok(paths)
}

pub fn generate_for_map(
    map_path: &Path,
    cfg: &Config,
    out_dir: &Path,
    options: &GenerateOptions,
) -> Result<(Vec<DatasetSample>, MapDatasetEntry), Box<dyn std::error::Error + Send + Sync>> {
    let map_path_str = map_path.to_string_lossy().to_string();
    let map = Map::from_path(&map_path_str);
    let (width, height) = map.dimensions();
    if width != cfg.map_size || height != cfg.map_size {
        return Err(format!(
            "expected {}x{} map, got {}x{} for {}",
            cfg.map_size,
            cfg.map_size,
            width,
            height,
            map_path.display()
        )
        .into());
    }

    fs::create_dir_all(out_dir)?;

    let open = open_cells(&map);
    let samples: Vec<DatasetSample> = open
        .par_iter()
        .map(|&(x, y)| {
            let mut rng = StdRng::seed_from_u64(((y as u64) << 32) ^ (x as u64) ^ 0x5EED5EED);
            DatasetSample {
                cell: (x, y),
                score: ambiguity_score((x, y), &map, cfg, &mut rng),
                crop: default_crop_window(&map, (x, y)),
            }
        })
        .collect();

    write_per_map_binary_outputs(out_dir, &samples)?;
    write_per_map_manifest(out_dir, &samples)?;
    write_score_grid(out_dir, &map, &samples)?;

    if options.write_previews {
        write_map_preview(out_dir, &map)?;
        write_score_heatmap(out_dir, &map, &samples)?;
        write_smoothed_score_heatmap(out_dir, &map, &samples)?;
        write_crop_preview(out_dir, &samples)?;
        write_summary(out_dir, &map_path_str, &samples, (width, height))?;
    }

    let (min_score, mean_score, max_score) = score_stats(&samples);
    let map_id = map_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("map")
        .to_string();

    let entry = MapDatasetEntry {
        map_id: map_id.clone(),
        map_path: map_path_str,
        rel_dir: map_id.clone(),
        sample_count: samples.len(),
        crop_offset: 0,
        score_offset: 0,
        score_grid_path: format!("{map_id}/scores_grid.bin"),
        min_score,
        mean_score,
        max_score,
    };

    Ok((samples, entry))
}

pub fn write_per_map_manifest(
    out_dir: &Path,
    samples: &[DatasetSample],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let path = out_dir.join("manifest.jsonl");
    let mut file = BufWriter::new(File::create(path)?);
    for (local_index, sample) in samples.iter().enumerate() {
        let line = format!(
            "{{\"x\":{},\"y\":{},\"score\":{:.6},\"local_index\":{}}}\n",
            sample.cell.0,
            sample.cell.1,
            sample.score,
            local_index
        );
        file.write_all(line.as_bytes())?;
    }
    file.flush()?;
    Ok(())
}

pub fn append_global_manifest(
    global_manifest: &Path,
    map_id: &str,
    per_map_manifest: &Path,
    global_start_index: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let text = fs::read_to_string(per_map_manifest)?;
    let mut file = BufWriter::new(
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(global_manifest)?,
    );

    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let local_index = extract_json_usize(line, "local_index")?;
        let x = extract_json_usize(line, "x")?;
        let y = extract_json_usize(line, "y")?;
        let score = extract_json_f32(line, "score")?;
        let merged = format!(
            "{{\"map_id\":\"{map_id}\",\"x\":{x},\"y\":{y},\"score\":{score:.6},\"crop_index\":{}}}\n",
            global_start_index + local_index
        );
        file.write_all(merged.as_bytes())?;
    }

    file.flush()?;
    Ok(())
}

fn extract_json_usize(line: &str, key: &str) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    let needle = format!("\"{key}\":");
    let tail = line
        .split(&needle)
        .nth(1)
        .ok_or_else(|| format!("missing {key} in manifest line"))?;
    let value = tail
        .trim()
        .split(|c: char| !c.is_ascii_digit())
        .next()
        .ok_or_else(|| format!("invalid {key} in manifest line"))?;
    Ok(value.parse()?)
}

fn extract_json_f32(line: &str, key: &str) -> Result<f32, Box<dyn std::error::Error + Send + Sync>> {
    let needle = format!("\"{key}\":");
    let tail = line
        .split(&needle)
        .nth(1)
        .ok_or_else(|| format!("missing {key} in manifest line"))?;
    let value = tail
        .trim()
        .trim_end_matches(',')
        .split(|c: char| c == ',' || c == '}')
        .next()
        .ok_or_else(|| format!("invalid {key} in manifest line"))?;
    Ok(value.parse()?)
}

pub fn write_dataset_index(
    path: &Path,
    cfg: &Config,
    maps: &[MapDatasetEntry],
    n_samples: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let index = DatasetIndex {
        version: 1,
        crop_size: CROP_SIZE,
        map_size: cfg.map_size,
        crop_bytes: CROP_SIZE * CROP_SIZE,
        score_dtype: "f32_le",
        crop_encoding: "u8 wall=255 open=0",
        wall_score_sentinel: WALL_SCORE_SENTINEL,
        n_maps: maps.len(),
        n_samples,
        crops_file: "crops.bin".to_string(),
        scores_file: "scores.bin".to_string(),
        manifest_file: "manifest.jsonl".to_string(),
        maps: maps.to_vec(),
    };

    let json = serde_json::to_string_pretty(&index)?;
    fs::write(path, json)?;
    Ok(())
}

pub fn write_training_readme(path: &Path) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let text = r#"Paraloc training dataset export

Global files (batch mode):
  index.json       - dataset metadata and per-map byte offsets
  manifest.jsonl   - one JSON object per sample (map_id, x, y, score, crop_index)
  crops.bin        - concatenated 64x64 crops (4096 bytes each, u8 wall=255 open=0)
  scores.bin       - concatenated f32 little-endian ambiguity scores
  TRAINING.txt     - this file

Per-map directory (<map_id>/):
  scores_grid.bin  - full 128x128 heatmap (primary label for U-Net training)
  crops.bin        - per-map crop archive
  scores.bin       - per-map score archive
  manifest.jsonl   - per-map sample manifest

scores_grid.bin layout:
  map_size * map_size float32 values in row-major order (x fastest).
  Wall cells use sentinel -1.0; open cells store ambiguity score in [0, 1].

U-Net training (scripts/colab_train_full.py):
  - maps:     dataset_out/maps_200/<map_id>.png
  - labels:   dataset_out/maps_200_train/<map_id>/scores_grid.bin
  - split:    dataset_out/maps_200_train/index.json
"#;
    fs::write(path, text)?;
    Ok(())
}

fn score_stats(samples: &[DatasetSample]) -> (f32, f32, f32) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let mut min_score = f32::INFINITY;
    let mut max_score = f32::NEG_INFINITY;
    let mut total = 0.0_f32;

    for sample in samples {
        min_score = min_score.min(sample.score);
        max_score = max_score.max(sample.score);
        total += sample.score;
    }

    (min_score, total / samples.len() as f32, max_score)
}

fn write_per_map_binary_outputs(
    out_dir: &Path,
    samples: &[DatasetSample],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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

fn write_score_grid(
    out_dir: &Path,
    map: &Map,
    samples: &[DatasetSample],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (width, height) = map.dimensions();
    let mut grid = vec![WALL_SCORE_SENTINEL; width * height];

    for sample in samples {
        let idx = sample.cell.1 * width + sample.cell.0;
        grid[idx] = sample.score;
    }

    let mut writer = BufWriter::new(File::create(out_dir.join("scores_grid.bin"))?);
    for value in grid {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}
fn write_summary(
    out_dir: &Path,
    map_path: &str,
    samples: &[DatasetSample],
    map_size: (usize, usize),
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (min_score, mean_score, max_score) = score_stats(samples);
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

fn write_map_preview(out_dir: &Path, map: &Map) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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
    samples: &[DatasetSample],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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
    samples: &[DatasetSample],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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

fn build_score_grid(map: &Map, samples: &[DatasetSample]) -> (Vec<Option<f32>>, f32, f32) {
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
    samples: &[DatasetSample],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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


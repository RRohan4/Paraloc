use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

use paraloc::{
    config::Config,
    dataset::{
        append_global_manifest, collect_map_pngs, generate_for_map, output_dir_for_map,
        write_dataset_index, write_training_readme, GenerateOptions, MapDatasetEntry,
    },
};
use rayon::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = parse_args();
    let mut cfg = Config::load_or_default("config.toml");
    if let Some(path) = &args.output_dir {
        cfg.output_dir = path.clone();
    }

    if let Some(maps_dir) = &args.maps_dir {
        run_batch(&cfg, maps_dir, &args)?;
        return Ok(());
    }

    let map_path = args
        .map_path
        .ok_or("single-map mode requires --map (or use --maps-dir for batch)")?;
    let out_dir = output_dir_for_map(Path::new(&cfg.output_dir), Path::new(&map_path))?;
    let options = GenerateOptions {
        write_previews: !args.no_previews,
    };

    let (samples, _) = generate_for_map(Path::new(&map_path), &cfg, &out_dir, &options)?;
    println!("wrote dataset preview to {}", out_dir.display());
    println!("samples={}", samples.len());
    if options.write_previews {
        println!(
            "files: crops.bin, scores.bin, scores_grid.bin, manifest.jsonl, map_preview.png, score_heatmap.png, score_heatmap_smooth.png, crop_preview.png, crop_preview.txt, summary.txt"
        );
    } else {
        println!("files: crops.bin, scores.bin, scores_grid.bin, manifest.jsonl");
    }
    Ok(())
}

struct CliArgs {
    map_path: Option<String>,
    maps_dir: Option<String>,
    output_dir: Option<String>,
    no_previews: bool,
}

fn parse_args() -> CliArgs {
    let mut map_path = None;
    let mut maps_dir = None;
    let mut output_dir = None;
    let mut no_previews = false;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--map" => map_path = Some(args.next().expect("missing value for --map")),
            "--maps-dir" => maps_dir = Some(args.next().expect("missing value for --maps-dir")),
            "--output-dir" => {
                output_dir = Some(args.next().expect("missing value for --output-dir"))
            }
            "--no-previews" => no_previews = true,
            other => panic!("unknown arg: {other}"),
        }
    }

    CliArgs {
        map_path,
        maps_dir,
        output_dir,
        no_previews,
    }
}

fn run_batch(
    cfg: &Config,
    maps_dir: &str,
    args: &CliArgs,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let maps_path = Path::new(maps_dir);
    let map_paths = collect_map_pngs(maps_path)?;
    if map_paths.is_empty() {
        return Err(format!("no PNG maps found in {}", maps_path.display()).into());
    }

    let export_root = PathBuf::from(&cfg.output_dir);
    fs::create_dir_all(&export_root)?;

    let crops_path = export_root.join("crops.bin");
    let scores_path = export_root.join("scores.bin");
    let manifest_path = export_root.join("manifest.jsonl");
    let _ = fs::remove_file(&crops_path);
    let _ = fs::remove_file(&scores_path);
    let _ = fs::remove_file(&manifest_path);

    let options = GenerateOptions {
        write_previews: !args.no_previews,
    };

    let started = Instant::now();
    println!(
        "processing {} maps from {} -> {}",
        map_paths.len(),
        maps_path.display(),
        export_root.display()
    );

    let per_map: Vec<(PathBuf, MapDatasetEntry)> = map_paths
        .par_iter()
        .map(
            |map_path| -> Result<(PathBuf, MapDatasetEntry), Box<dyn std::error::Error + Send + Sync>> {
                let out_dir = output_dir_for_map(&export_root, map_path)?;
                let (_, entry) = generate_for_map(map_path, cfg, &out_dir, &options)?;
                println!("  finished {}", entry.map_id);
                Ok((out_dir, entry))
            },
        )
        .collect::<Result<Vec<_>, _>>()?;

    let crop_stride = (paraloc::dataset::CROP_SIZE * paraloc::dataset::CROP_SIZE) as u64;
    let score_stride = std::mem::size_of::<f32>() as u64;
    let mut entries = Vec::with_capacity(per_map.len());
    let mut global_index = 0_usize;
    let mut crop_offset = 0_u64;
    let mut score_offset = 0_u64;
    let mut n_samples = 0_usize;

    for (out_dir, mut entry) in per_map {
        let per_map_crops = fs::read(out_dir.join("crops.bin"))?;
        let per_map_scores = fs::read(out_dir.join("scores.bin"))?;
        let sample_count = entry.sample_count;
        if per_map_crops.len() != sample_count * crop_stride as usize {
            return Err(format!(
                "crop size mismatch for {}: expected {} bytes, got {}",
                entry.map_id,
                sample_count * crop_stride as usize,
                per_map_crops.len()
            )
            .into());
        }
        if per_map_scores.len() != sample_count * score_stride as usize {
            return Err(format!(
                "score size mismatch for {}: expected {} bytes, got {}",
                entry.map_id,
                sample_count * score_stride as usize,
                per_map_scores.len()
            )
            .into());
        }

        entry.crop_offset = crop_offset;
        entry.score_offset = score_offset;
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&crops_path)?
            .write_all(&per_map_crops)?;
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&scores_path)?
            .write_all(&per_map_scores)?;
        append_global_manifest(
            &manifest_path,
            &entry.map_id,
            &out_dir.join("manifest.jsonl"),
            global_index,
        )?;
        global_index += sample_count;
        crop_offset += crop_stride * sample_count as u64;
        score_offset += score_stride * sample_count as u64;
        n_samples += sample_count;
        entries.push(entry);
    }

    write_dataset_index(&export_root.join("index.json"), cfg, &entries, n_samples)?;
    write_training_readme(&export_root.join("TRAINING.txt"))?;

    println!(
        "batch complete: {} maps, {} samples, {:.1}s",
        entries.len(),
        n_samples,
        started.elapsed().as_secs_f64()
    );
    println!("export root: {}", export_root.display());
    println!("global files: index.json, manifest.jsonl, crops.bin, scores.bin, TRAINING.txt");
    Ok(())
}

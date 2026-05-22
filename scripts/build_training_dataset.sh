#!/usr/bin/env bash
# Process every map in dataset_out/maps_200 into ML-ready training files.
# Maps stay in maps_200; labels/crops go to dataset_out/maps_200_train.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MAPS_DIR="${MAPS_DIR:-dataset_out/maps_200}"
OUT_DIR="${OUT_DIR:-dataset_out/maps_200_train}"

echo "maps:   $MAPS_DIR"
echo "output: $OUT_DIR"
echo "(rebuilds index.json, global crops.bin/scores.bin/manifest.jsonl, per-map scores_grid.bin)"
echo "(add --no-previews to skip PNG previews; omit flag for portfolio heatmap PNGs)"
echo

cargo build --release --bin gen_dataset

cargo run --release --bin gen_dataset -- \
  --maps-dir "$MAPS_DIR" \
  --output-dir "$OUT_DIR" \
  --no-previews

echo
echo "Done. Training bundle:"
echo "  $OUT_DIR/index.json"
echo "  $OUT_DIR/manifest.jsonl"
echo "  $OUT_DIR/crops.bin"
echo "  $OUT_DIR/scores.bin"
echo "  $OUT_DIR/TRAINING.txt"
echo "Per-map bins: $OUT_DIR/map_XXX/"

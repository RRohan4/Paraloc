#!/usr/bin/env python3
"""
Run full-map ambiguity heatmap inference with the UNet128 checkpoint from
colab_train_full.py.

Example:
  python scripts/infer_heatmap.py \
    --checkpoint heatmap_model_full.pt \
    --map dataset_out/maps_200/map_129.png \
    --output pred_map_129.png

With ground-truth comparison:
  python scripts/infer_heatmap.py \
    --checkpoint heatmap_model_full.pt \
    --map dataset_out/maps_200/map_129.png \
    --ground-truth dataset_out/maps_200_train/map_129/scores_grid.bin \
    --compare-out compare_map_129.png
"""

from __future__ import annotations

# Per-map Rust Monte Carlo sim time at n_score_orientations=16 (default scoring config).
DEFAULT_SIM_SECONDS = 126.0
# Matches SMOOTH_SIGMA_TILES in src/dataset.rs (score_heatmap_smooth.png).
SMOOTH_SIGMA_TILES = 0.9

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from colab_train_full import MAP_SIZE, UNet128, load_occupancy, load_score_grid


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_model(checkpoint: Path, device: torch.device) -> UNet128:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "UNet128")
    if arch != "UNet128":
        raise ValueError(f"expected arch=UNet128, got {arch!r}")

    model = UNet128().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def predict_heatmap(
    model: UNet128,
    map_png: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    occ = load_occupancy(map_png)
    if occ.shape != (MAP_SIZE, MAP_SIZE):
        raise ValueError(f"expected {MAP_SIZE}x{MAP_SIZE} map, got {occ.shape}")

    x = torch.from_numpy(occ[None, None]).float().to(device)
    pred = model(x)[0, 0].cpu().numpy()
    mask = occ > 0.5
    return pred, occ, mask


def benchmark_inference(
    model: UNet128,
    map_png: Path,
    device: torch.device,
    *,
    warmup: int = 3,
    repeats: int = 10,
) -> float:
    for _ in range(warmup):
        predict_heatmap(model, map_png, device)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict_heatmap(model, map_png, device)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def masked_stats(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    open_pred = pred[mask]
    open_gt = gt[mask]
    if open_gt.size <= 1:
        return 0.0
    return float(np.corrcoef(open_gt, open_pred)[0, 1])


def format_duration(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.2f}s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.1f}ms"
    return f"{seconds * 1e6:.0f}µs"


def format_speedup(sim_seconds: float, infer_seconds: float) -> str:
    if infer_seconds <= 0:
        return "n/a"
    ratio = sim_seconds / infer_seconds
    if ratio >= 100:
        return f"~{ratio:.0f}× faster"
    if ratio >= 10:
        return f"~{ratio:.0f}× faster"
    return f"~{ratio:.1f}× faster"


def smooth_score_grid(
    scores: np.ndarray,
    mask: np.ndarray,
    *,
    sigma: float = SMOOTH_SIGMA_TILES,
) -> np.ndarray:
    """Wall-aware Gaussian blur matching Rust smooth_score_grid in dataset.rs."""
    height, width = scores.shape
    kernel_radius = int(np.ceil(sigma * 3.0))
    smoothed = np.zeros_like(scores, dtype=np.float32)

    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue

            weighted = 0.0
            total_weight = 0.0
            y_min = max(0, y - kernel_radius)
            y_max = min(height - 1, y + kernel_radius)
            x_min = max(0, x - kernel_radius)
            x_max = min(width - 1, x + kernel_radius)

            for ny in range(y_min, y_max + 1):
                for nx in range(x_min, x_max + 1):
                    if not mask[ny, nx]:
                        continue
                    dx = float(x - nx)
                    dy = float(y - ny)
                    dist_sq = dx * dx + dy * dy
                    weight = np.exp(-dist_sq / (2.0 * sigma * sigma))
                    weighted += float(scores[ny, nx]) * weight
                    total_weight += weight

            if total_weight > 0.0:
                smoothed[y, x] = weighted / total_weight

    return smoothed


def heatmap_to_rgb(
    scores: np.ndarray,
    mask: np.ndarray,
    *,
    cmap_name: str = "inferno",
    vmin: float = 0.0,
    vmax: float = 1.0,
    wall_rgb: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    cmap = plt.colormaps[cmap_name]
    norm = np.clip((scores - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
    rgba = cmap(norm)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    rgb[~mask] = wall_rgb
    return rgb


def save_heatmap_png(
    scores: np.ndarray,
    mask: np.ndarray,
    output: Path,
    *,
    cmap_name: str = "inferno",
    scale: int = 4,
) -> None:
    rgb = heatmap_to_rgb(scores, mask, cmap_name=cmap_name)
    img = Image.fromarray(rgb, mode="RGB")
    if scale > 1:
        img = img.resize((MAP_SIZE * scale, MAP_SIZE * scale), resample=Image.NEAREST)
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)


def save_comparison_figure(
    pred: np.ndarray,
    mask: np.ndarray,
    output: Path,
    *,
    ground_truth: np.ndarray,
    corr: float,
    infer_seconds: float,
    sim_seconds: float,
) -> None:
    speedup = format_speedup(sim_seconds, infer_seconds)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(
        f"Rust sim {format_duration(sim_seconds)}  →  UNet {format_duration(infer_seconds)}  ({speedup})",
        fontsize=11,
        y=1.02,
    )

    axes[0].imshow(np.where(mask, ground_truth, np.nan), vmin=0, vmax=1, cmap="inferno")
    axes[0].set_title("ground truth (Rust sim labels)")
    axes[1].imshow(np.where(mask, pred, np.nan), vmin=0, vmax=1, cmap="inferno")
    axes[1].set_title(f"UNet prediction  corr={corr:.2f}")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a full-map ambiguity heatmap.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("heatmap_model_full.pt"),
        help="UNet128 checkpoint from colab_train_full.py",
    )
    parser.add_argument("--map", type=Path, required=True, help="128x128 occupancy PNG")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Standalone inferno heatmap PNG (default: <map_stem>_pred.png)",
    )
    parser.add_argument(
        "--compare-out",
        type=Path,
        default=None,
        help="Optional ground truth vs prediction figure PNG (requires --ground-truth)",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Optional scores_grid.bin for side-by-side comparison",
    )
    parser.add_argument(
        "--sim-seconds",
        type=float,
        default=DEFAULT_SIM_SECONDS,
        help="Reference Rust sim wall time per map for speedup label (default: 126s at 16 orientations)",
    )
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor for --output PNG")
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Skip display-only Gaussian smoothing (metrics always use raw scores)",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--show", action="store_true", help="Display the comparison figure")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"device={device}")

    if not args.checkpoint.exists():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")
    if not args.map.exists():
        raise SystemExit(f"missing map: {args.map}")

    model = load_model(args.checkpoint, device)
    pred, _, mask = predict_heatmap(model, args.map, device)
    infer_seconds = benchmark_inference(model, args.map, device)

    pred_display = pred if args.no_smooth else smooth_score_grid(pred, mask)

    output = args.output or args.map.with_name(f"{args.map.stem}_pred.png")
    save_heatmap_png(pred_display, mask, output, scale=args.scale)
    print(f"saved {output}")
    print(
        f"inference={format_duration(infer_seconds)}  "
        f"rust_sim={format_duration(args.sim_seconds)}  "
        f"speedup={format_speedup(args.sim_seconds, infer_seconds)}"
    )

    gt = None
    corr = None
    if args.ground_truth is not None:
        if not args.ground_truth.exists():
            raise SystemExit(f"missing ground truth: {args.ground_truth}")
        gt, gt_mask = load_score_grid(args.ground_truth)
        mask = mask & gt_mask.astype(bool)
        corr = masked_stats(pred, gt, mask)
        print(f"corr={corr:.4f}")

    compare_out = args.compare_out
    if compare_out is None and args.show:
        compare_out = args.map.with_name(f"{args.map.stem}_compare.png")

    if compare_out is not None:
        if gt is None:
            raise SystemExit("--compare-out requires --ground-truth")
        gt_display = gt if args.no_smooth else smooth_score_grid(gt, mask)
        save_comparison_figure(
            pred_display,
            mask,
            compare_out,
            ground_truth=gt_display,
            corr=corr or 0.0,
            infer_seconds=infer_seconds,
            sim_seconds=args.sim_seconds,
        )
        print(f"saved {compare_out}")
        if args.show:
            img = Image.open(compare_out)
            plt.figure(figsize=(9, 4))
            plt.imshow(img)
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    main()

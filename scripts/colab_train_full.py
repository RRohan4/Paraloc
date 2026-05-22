#!/usr/bin/env python3
"""
Paraloc — full Colab training + diagnostics (paste or run as one file).

Why epoch 1 already has "low" val loss:
  - Scores are mostly high (~0.7–0.9 mean). Predicting the dataset mean
    already gets MSE ~0.02–0.04 before the net learns any geometry.
  - Watch correlation & side-by-side heatmaps, not MSE alone.

Upload to Drive (data only):
  dataset_out/maps_200/
  dataset_out/maps_200_train/

In Colab: Runtime -> GPU, mount Drive, then:
  %run /content/drive/MyDrive/colab_train_full.py
or paste this entire file into one cell.
"""

from __future__ import annotations

import json
import random
import struct
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ============ EDIT THESE PATHS ============
ROOT = Path("/content/drive/MyDrive")
TRAIN_ROOT = ROOT / "dataset_out/maps_200_train"
MAPS_DIR = ROOT / "dataset_out/maps_200"
OUT_CKPT = ROOT / "dataset_out/heatmap_model_full.pt"
OUT_PLOT = ROOT / "dataset_out/training_curves.png"
# =========================================

MAP_SIZE = 128
WALL_SENTINEL = -1.0
EPOCHS = 60
BATCH_SIZE = 16
LR = 1e-3
VAL_FRACTION = 0.1
SEED = 42
NUM_VIZ_MAPS = 4


def load_occupancy(map_png: Path) -> np.ndarray:
    arr = np.array(Image.open(map_png).convert("L"), dtype=np.float32)
    return (arr >= 128).astype(np.float32)


def load_score_grid(grid_path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = grid_path.read_bytes()
    n = MAP_SIZE * MAP_SIZE
    scores = np.array(struct.unpack(f"<{n}f", raw), dtype=np.float32).reshape(MAP_SIZE, MAP_SIZE)
    mask = scores != WALL_SENTINEL
    target = np.where(mask, scores, 0.0).astype(np.float32)
    return target, mask.astype(np.float32)


class MapHeatmapDataset(Dataset):
    def __init__(self, entries: list[dict], maps_dir: Path, train_root: Path):
        self.entries = entries
        self.maps_dir = maps_dir
        self.train_root = train_root

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        entry = self.entries[idx]
        map_id = entry["map_id"]
        occ = load_occupancy(self.maps_dir / f"{map_id}.png")
        target, mask = load_score_grid(self.train_root / map_id / "scores_grid.bin")
        return (
            torch.from_numpy(occ[None]),
            torch.from_numpy(target[None]),
            torch.from_numpy(mask[None]),
            map_id,
        )


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet128(nn.Module):
    """Proper-ish U-Net for 128x128 -> 128x128 (stronger than TinyUNet)."""

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) ** 2
    return (diff * mask).sum() / mask.sum().clamp_min(1.0)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs()
    return (diff * mask).sum() / mask.sum().clamp_min(1.0)


def split_maps(index_path: Path, val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    maps = list(json.loads(index_path.read_text())["maps"])
    rng = random.Random(seed)
    rng.shuffle(maps)
    n_val = max(1, int(len(maps) * val_fraction))
    return maps[n_val:], maps[:n_val]


def dataset_label_stats(entries: list[dict], train_root: Path) -> dict[str, float]:
    """Mean score on open cells — the 'predict mean' baseline."""
    values = []
    for e in entries:
        target, mask = load_score_grid(train_root / e["map_id"] / "scores_grid.bin")
        values.append(target[mask.astype(bool)])
    allv = np.concatenate(values)
    return {
        "mean": float(allv.mean()),
        "std": float(allv.std()),
        "min": float(allv.min()),
        "max": float(allv.max()),
    }


def baseline_mean_mse(entries: list[dict], train_root: Path, mean_score: float) -> float:
    err = 0.0
    count = 0.0
    for e in entries:
        target, mask = load_score_grid(train_root / e["map_id"] / "scores_grid.bin")
        open_t = target[mask.astype(bool)]
        err += ((open_t - mean_score) ** 2).sum()
        count += open_t.size
    return float(err / count)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    n = 0
    preds_all = []
    gts_all = []

    for x, y, m, _ in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        pred = model(x)
        bs = x.size(0)
        mse_sum += masked_mse(pred, y, m).item() * bs
        mae_sum += masked_mae(pred, y, m).item() * bs
        n += bs

        for i in range(bs):
            mask_i = m[i, 0].cpu().numpy().astype(bool)
            preds_all.append(pred[i, 0].cpu().numpy()[mask_i])
            gts_all.append(y[i, 0].cpu().numpy()[mask_i])

    pred_cat = np.concatenate(preds_all)
    gt_cat = np.concatenate(gts_all)
    corr = float(np.corrcoef(pred_cat, gt_cat)[0, 1]) if pred_cat.size > 1 else 0.0

    return {
        "mse": mse_sum / n,
        "mae": mae_sum / n,
        "corr": corr,
        "pred_mean": float(pred_cat.mean()),
        "gt_mean": float(gt_cat.mean()),
    }


def plot_training_curves(history: dict[str, list[float]], baseline_val_mse: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    epochs = range(1, len(history["train_mse"]) + 1)

    axes[0].plot(epochs, history["train_mse"], label="train MSE")
    axes[0].plot(epochs, history["val_mse"], label="val MSE")
    axes[0].axhline(baseline_val_mse, color="red", linestyle="--", label="val mean-only baseline")
    axes[0].set_title("MSE (lower = closer to labels)")
    axes[0].legend()

    axes[1].plot(epochs, history["val_corr"], label="val correlation", color="green")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Correlation (higher = better structure)")

    axes[2].plot(epochs, history["val_mae"], label="val MAE")
    axes[2].set_title("MAE on open cells")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120)
    plt.show()
    print(f"saved {OUT_PLOT}")


@torch.no_grad()
def visualize_maps(
    model: nn.Module,
    val_entries: list[dict],
    device: torch.device,
    n: int,
) -> None:
    model.eval()
    rng = random.Random(0)
    picks = val_entries if len(val_entries) <= n else rng.sample(val_entries, n)

    fig, axes = plt.subplots(len(picks), 4, figsize=(14, 3.5 * len(picks)))
    if len(picks) == 1:
        axes = np.expand_dims(axes, 0)

    for row, entry in enumerate(picks):
        mid = entry["map_id"]
        occ = load_occupancy(MAPS_DIR / f"{mid}.png")
        gt, mask = load_score_grid(TRAIN_ROOT / mid / "scores_grid.bin")
        pred = model(torch.from_numpy(occ[None, None]).float().to(device))[0, 0].cpu().numpy()

        open_gt = gt[mask.astype(bool)]
        open_pred = pred[mask.astype(bool)]
        corr = np.corrcoef(open_gt, open_pred)[0, 1] if open_gt.size > 1 else 0.0
        err = np.abs(open_pred - open_gt).mean()

        axes[row, 0].imshow(occ, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{mid} map")
        axes[row, 1].imshow(np.where(mask, gt, np.nan), vmin=0, vmax=1, cmap="inferno")
        axes[row, 1].set_title("ground truth")
        axes[row, 2].imshow(np.where(mask, pred, np.nan), vmin=0, vmax=1, cmap="inferno")
        axes[row, 2].set_title(f"pred  corr={corr:.2f}")
        axes[row, 3].imshow(np.where(mask, np.abs(pred - gt), np.nan), vmin=0, vmax=0.5, cmap="magma")
        axes[row, 3].set_title(f"|error| mae={err:.3f}")

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    assert (TRAIN_ROOT / "index.json").exists(), f"missing {TRAIN_ROOT / 'index.json'}"
    assert MAPS_DIR.exists(), f"missing {MAPS_DIR}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(
        "Score meaning: 0 = easy to localize, 1 = very ambiguous. "
        "Low MSE on epoch 1 often means 'predict ~mean score', not a great model."
    )

    train_maps, val_maps = split_maps(TRAIN_ROOT / "index.json", VAL_FRACTION, SEED)
    train_stats = dataset_label_stats(train_maps, TRAIN_ROOT)
    val_stats = dataset_label_stats(val_maps, TRAIN_ROOT)
    mean_baseline_val_mse = baseline_mean_mse(val_maps, TRAIN_ROOT, train_stats["mean"])

    print(f"train maps={len(train_maps)} val maps={len(val_maps)}")
    print(f"train label stats (open cells): {train_stats}")
    print(f"val   label stats (open cells): {val_stats}")
    print(
        f"VAL MSE if we always predict train mean ({train_stats['mean']:.3f}): "
        f"{mean_baseline_val_mse:.6f}  <- compare to epoch 1 val MSE"
    )

    train_loader = DataLoader(
        MapHeatmapDataset(train_maps, MAPS_DIR, TRAIN_ROOT),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        MapHeatmapDataset(val_maps, MAPS_DIR, TRAIN_ROOT),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = UNet128().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    history: dict[str, list[float]] = {
        "train_mse": [],
        "val_mse": [],
        "val_mae": [],
        "val_corr": [],
    }

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_mse_acc = 0.0
        n_batches = 0
        for x, y, m, _ in train_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred = model(x)
            loss = masked_mse(pred, y, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_mse_acc += loss.item()
            n_batches += 1

        train_mse = train_mse_acc / max(n_batches, 1)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["mse"])

        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_metrics["mse"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_corr"].append(val_metrics["corr"])

        print(
            f"epoch {epoch:03d}/{EPOCHS}  "
            f"train_mse={train_mse:.6f}  val_mse={val_metrics['mse']:.6f}  "
            f"val_mae={val_metrics['mae']:.6f}  val_corr={val_metrics['corr']:.3f}  "
            f"pred_mean={val_metrics['pred_mean']:.3f} gt_mean={val_metrics['gt_mean']:.3f}"
        )

    elapsed = time.time() - t0
    print(f"training done in {elapsed/60:.1f} min")

    OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "arch": "UNet128",
            "map_size": MAP_SIZE,
            "history": history,
            "train_label_stats": train_stats,
        },
        OUT_CKPT,
    )
    print(f"saved {OUT_CKPT}")

    plot_training_curves(history, mean_baseline_val_mse)
    visualize_maps(model, val_maps, device, NUM_VIZ_MAPS)


if __name__ == "__main__":
    main()

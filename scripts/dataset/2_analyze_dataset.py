"""
Step 2 — Dataset Statistics & Visual Analysis
===============================================
Computes and plots:
  - Bounding box size distribution (width, height, area)
  - Box center heatmap (where on the image bags tend to appear)
  - Aspect ratio distribution
  - Boxes-per-image histogram
  - Sample grid with drawn annotations

Outputs saved to:  preprocess/analysis/

Run:
    python preprocess/2_analyze_dataset.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
from tqdm import tqdm

DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset" / "flour_dataset"
OUT_DIR      = Path(__file__).resolve().parent / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_VIZ   = 16   # number of images to draw in the sample grid
HEATMAP_RES  = 100  # heatmap grid resolution


def load_labels(split: str):
    """Return list of (cx, cy, w, h) arrays, one per image."""
    lbl_dir = DATASET_ROOT / "labels" / split
    all_boxes = []
    per_image_counts = []
    for txt in tqdm(sorted(lbl_dir.glob("*.txt")), desc=f"  Loading {split} labels"):
        lines = txt.read_text().splitlines()
        boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                _, cx, cy, w, h = parts
                boxes.append([float(cx), float(cy), float(w), float(h)])
        per_image_counts.append(len(boxes))
        all_boxes.extend(boxes)
    return np.array(all_boxes, dtype=np.float32), per_image_counts


def plot_box_distributions(boxes: np.ndarray, split: str):
    if len(boxes) == 0:
        print(f"  [WARN] No boxes found for split '{split}', skipping distribution plot.")
        return

    w, h    = boxes[:, 2], boxes[:, 3]
    cx, cy  = boxes[:, 0], boxes[:, 1]
    area    = w * h
    aspect  = w / (h + 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Bounding Box Statistics — {split} ({len(boxes):,} boxes)", fontsize=14)

    axes[0, 0].hist(w,      bins=80, color="steelblue", edgecolor="none")
    axes[0, 0].set(title="Box Width (normalized)",  xlabel="width",  ylabel="count")

    axes[0, 1].hist(h,      bins=80, color="coral",     edgecolor="none")
    axes[0, 1].set(title="Box Height (normalized)", xlabel="height", ylabel="count")

    axes[0, 2].hist(area,   bins=80, color="green",     edgecolor="none")
    axes[0, 2].set(title="Box Area (w×h)",          xlabel="area",   ylabel="count")

    axes[1, 0].hist(aspect, bins=80, color="purple",    edgecolor="none")
    axes[1, 0].axvline(1.0, color="red", linestyle="--", label="square")
    axes[1, 0].set(title="Aspect Ratio (w/h)",      xlabel="ratio",  ylabel="count")
    axes[1, 0].legend()

    axes[1, 1].scatter(cx, cy, s=1, alpha=0.05, color="navy")
    axes[1, 1].set(title="Box Center Distribution", xlabel="cx", ylabel="cy",
                   xlim=(0, 1), ylim=(1, 0))
    axes[1, 1].set_aspect("equal")

    # Center heatmap
    heatmap, xedges, yedges = np.histogram2d(cx, cy, bins=HEATMAP_RES,
                                              range=[[0, 1], [0, 1]])
    im = axes[1, 2].imshow(heatmap.T, origin="upper", extent=[0, 1, 0, 1],
                           cmap="hot", aspect="auto")
    axes[1, 2].set(title="Box Center Heatmap", xlabel="cx", ylabel="cy")
    plt.colorbar(im, ax=axes[1, 2])

    plt.tight_layout()
    out = OUT_DIR / f"{split}_box_stats.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved: {out}")

    # Print numeric summary
    print(f"\n  [{split.upper()}] Box Statistics:")
    print(f"    Total boxes : {len(boxes):,}")
    print(f"    Width  — mean: {w.mean():.4f}  std: {w.std():.4f}  min: {w.min():.4f}  max: {w.max():.4f}")
    print(f"    Height — mean: {h.mean():.4f}  std: {h.std():.4f}  min: {h.min():.4f}  max: {h.max():.4f}")
    print(f"    Area   — mean: {area.mean():.4f}  median: {np.median(area):.4f}")
    print(f"    Aspect — mean: {aspect.mean():.4f}  (>1 = wider than tall)")

    small  = (area < 0.01).sum()
    medium = ((area >= 0.01) & (area < 0.09)).sum()
    large  = (area >= 0.09).sum()
    print(f"    Size buckets: small(<1%): {small:,}  medium(1–9%): {medium:,}  large(>9%): {large:,}")


def plot_boxes_per_image(counts: list, split: str):
    arr = np.array(counts)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(arr, bins=range(0, min(int(arr.max()) + 2, 60)), color="teal", edgecolor="white")
    ax.set(title=f"Boxes per Image — {split}",
           xlabel="Number of bags", ylabel="Number of images")
    ax.axvline(arr.mean(), color="red", linestyle="--", label=f"mean={arr.mean():.1f}")
    ax.legend()
    out = OUT_DIR / f"{split}_boxes_per_image.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved: {out}")
    print(f"    Boxes/image — mean: {arr.mean():.1f}  max: {arr.max()}  "
          f"images with 0 boxes: {(arr == 0).sum()}")


def plot_sample_grid(split: str, n: int = SAMPLE_VIZ):
    img_dir = DATASET_ROOT / "images" / split
    lbl_dir = DATASET_ROOT / "labels" / split
    all_imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    sample   = random.sample(all_imgs, min(n, len(all_imgs)))

    cols = 4
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for ax, img_path in zip(axes, sample):
        img = Image.open(img_path).convert("RGB")
        iw, ih = img.size
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(img_path.stem[:20], fontsize=7)

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts)
                x1 = (cx - bw / 2) * iw
                y1 = (cy - bh / 2) * ih
                rect = patches.Rectangle(
                    (x1, y1), bw * iw, bh * ih,
                    linewidth=1.5, edgecolor="lime", facecolor="none"
                )
                ax.add_patch(rect)

    fig.suptitle(f"Sample Annotations — {split}", fontsize=13)
    plt.tight_layout()
    out = OUT_DIR / f"{split}_sample_grid.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  Saved: {out}")


def main():
    print("=" * 60)
    print("Dataset Analysis")
    print(f"Output dir: {OUT_DIR}")
    print("=" * 60)

    for split in ("train", "val"):
        lbl_dir = DATASET_ROOT / "labels" / split
        if not lbl_dir.exists():
            print(f"\n[SKIP] {split}: labels directory not found yet.")
            continue

        print(f"\n[{split.upper()}]")
        boxes, counts = load_labels(split)
        plot_box_distributions(boxes, split)
        plot_boxes_per_image(counts, split)
        plot_sample_grid(split)

    print("\nDone. Review PNGs in:", OUT_DIR)


if __name__ == "__main__":
    main()

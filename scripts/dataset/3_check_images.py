"""
Step 3 — Image Health Check
=============================
Scans every image in train/ and val/ for:
  - Corrupted / unreadable files
  - Truncated JPEG/PNG data
  - Extreme aspect ratios (likely wrong orientation)
  - Very small images that may hurt training (< 64px on any side)
  - Duplicate detection via file-size hash bucket (fast approximation)

Writes a report to preprocess/analysis/image_health.txt
and lists corrupt files to preprocess/analysis/corrupt_images.txt
  → You can then remove them with:  python preprocess/3_check_images.py --remove

Run:
    python preprocess/3_check_images.py
    python preprocess/3_check_images.py --remove   # actually delete bad files + their labels
"""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset" / "flour_dataset"
OUT_DIR      = Path(__file__).resolve().parent / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS       = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_DIM          = 64     # minimum acceptable width or height in pixels
MAX_ASPECT_RATIO = 10.0   # flag images wider/taller than 10:1


def check_split(split: str, remove: bool = False):
    img_dir = DATASET_ROOT / "images" / split
    lbl_dir = DATASET_ROOT / "labels" / split

    if not img_dir.exists():
        print(f"  [SKIP] {img_dir} does not exist.")
        return []

    all_imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    print(f"\n[{split.upper()}]  {len(all_imgs):,} images")

    corrupt       = []
    too_small     = []
    bad_aspect    = []
    size_buckets  = defaultdict(list)   # (file_size_bucket) → [paths]  for cheap dupe check
    widths, heights = [], []

    for img_path in tqdm(all_imgs, desc=f"  Checking {split}"):
        # --- Corruption check ---
        try:
            from PIL import ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = False   # strict mode
            with Image.open(img_path) as im:
                im.verify()                           # detects truncation
        except (UnidentifiedImageError, Exception) as e:
            corrupt.append((img_path, str(e)))
            continue

        # Re-open (verify() closes the file) to get dimensions
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True    # lenient for size reading
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            corrupt.append((img_path, "cannot read dimensions"))
            continue

        widths.append(w)
        heights.append(h)

        if min(w, h) < MIN_DIM:
            too_small.append((img_path, w, h))

        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > MAX_ASPECT_RATIO:
            bad_aspect.append((img_path, w, h, round(aspect, 1)))

        # Bucket by rounded file-size (within ±5 KB) for fast dupe detection
        fsize = (img_path.stat().st_size // 5120) * 5120
        size_buckets[fsize].append(img_path)

    # --- Duplicate candidates (same size bucket → hash to confirm) ---
    dupe_candidates = []
    for bucket, paths in size_buckets.items():
        if len(paths) < 2:
            continue
        hash_map = defaultdict(list)
        for p in paths:
            h = hashlib.md5(p.read_bytes()).hexdigest()
            hash_map[h].append(p)
        for h, dupes in hash_map.items():
            if len(dupes) > 1:
                dupe_candidates.append(dupes)

    import numpy as np
    wa, ha = np.array(widths), np.array(heights)

    print(f"  Resolution — mean: {wa.mean():.0f}×{ha.mean():.0f}  "
          f"min: {wa.min()}×{ha.min()}  max: {wa.max()}×{ha.max()}")
    print(f"  Corrupt         : {len(corrupt)}")
    print(f"  Too small (<{MIN_DIM}px) : {len(too_small)}")
    print(f"  Bad aspect (>{MAX_ASPECT_RATIO}:1) : {len(bad_aspect)}")
    print(f"  Exact duplicates: {sum(len(d)-1 for d in dupe_candidates)}")

    if remove and corrupt:
        print(f"\n  Removing {len(corrupt)} corrupt files…")
        for img_path, _ in corrupt:
            img_path.unlink(missing_ok=True)
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            lbl_path.unlink(missing_ok=True)
        print("  Done.")

    return corrupt, too_small, bad_aspect, dupe_candidates


def write_report(results: dict):
    report_path = OUT_DIR / "image_health.txt"
    corrupt_list = OUT_DIR / "corrupt_images.txt"

    all_corrupt = []
    lines = []
    for split, data in results.items():
        if not data:
            continue
        corrupt, too_small, bad_aspect, dupes = data
        lines.append(f"\n=== {split.upper()} ===")
        lines.append(f"Corrupt         : {len(corrupt)}")
        lines.append(f"Too small       : {len(too_small)}")
        lines.append(f"Bad aspect ratio: {len(bad_aspect)}")
        lines.append(f"Exact dupes     : {sum(len(d)-1 for d in dupes)}")

        if corrupt:
            lines.append("\nCorrupt files:")
            for p, err in corrupt:
                lines.append(f"  {p}  [{err}]")
                all_corrupt.append(str(p))

        if too_small:
            lines.append("\nToo-small images:")
            for p, w, h in too_small[:20]:
                lines.append(f"  {p}  [{w}×{h}]")

        if bad_aspect:
            lines.append("\nBad aspect ratio:")
            for p, w, h, ratio in bad_aspect[:20]:
                lines.append(f"  {p}  [{w}×{h}, ratio={ratio}]")

        if dupes:
            lines.append("\nDuplicate groups:")
            for group in dupes[:10]:
                lines.append(f"  keep: {group[0]}")
                for p in group[1:]:
                    lines.append(f"  drop: {p}")

    report_path.write_text("\n".join(lines))
    print(f"\n  Report  : {report_path}")

    if all_corrupt:
        corrupt_list.write_text("\n".join(all_corrupt))
        print(f"  Corrupt list: {corrupt_list}")
        print(f"  Re-run with --remove to delete corrupt files and their labels.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove", action="store_true",
                        help="Delete corrupt image files and their corresponding label files")
    args = parser.parse_args()

    print("=" * 60)
    print("Image Health Check")
    print("=" * 60)

    results = {}
    for split in ("train", "val"):
        data = check_split(split, remove=args.remove)
        if data:
            results[split] = data

    write_report(results)


if __name__ == "__main__":
    main()

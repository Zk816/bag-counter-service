"""
Step 1 — Dataset Integrity Validation
======================================
Checks every image has a matching label file (and vice versa),
validates every annotation line is well-formed, and reports
any orphans or broken entries.

Run:
    python preprocess/1_validate_dataset.py
"""

import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset" / "flour_dataset"
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NC           = 1   # number of classes


def validate_split(split: str) -> dict:
    img_dir   = DATASET_ROOT / "images"  / split
    lbl_dir   = DATASET_ROOT / "labels"  / split

    if not img_dir.exists():
        print(f"  [ERROR] Image directory not found: {img_dir}")
        return {}
    if not lbl_dir.exists():
        print(f"  [ERROR] Label directory not found: {lbl_dir}")
        return {}

    image_stems = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    label_stems = {p.stem for p in lbl_dir.iterdir() if p.suffix == ".txt"}

    missing_labels = sorted(image_stems - label_stems)
    orphan_labels  = sorted(label_stems - image_stems)

    print(f"\n[{split.upper()}]")
    print(f"  Images : {len(image_stems)}")
    print(f"  Labels : {len(label_stems)}")

    if missing_labels:
        print(f"  [WARN] {len(missing_labels)} images with no label file:")
        for s in missing_labels[:10]:
            print(f"         {s}.jpg")
        if len(missing_labels) > 10:
            print(f"         ... and {len(missing_labels) - 10} more")
    else:
        print("  [OK] Every image has a matching label file")

    if orphan_labels:
        print(f"  [WARN] {len(orphan_labels)} label files with no matching image:")
        for s in orphan_labels[:10]:
            print(f"         {s}.txt")
    else:
        print("  [OK] No orphan label files")

    # Validate annotation content
    bad_lines   = []
    empty_labels = []
    total_boxes = 0

    for txt_path in lbl_dir.glob("*.txt"):
        lines = txt_path.read_text().splitlines()
        if not lines or all(l.strip() == "" for l in lines):
            empty_labels.append(txt_path.name)
            continue
        for ln, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                bad_lines.append((txt_path.name, ln, f"expected 5 fields, got {len(parts)}"))
                continue
            try:
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
            except ValueError:
                bad_lines.append((txt_path.name, ln, "non-numeric field"))
                continue
            if cls_id < 0 or cls_id >= NC:
                bad_lines.append((txt_path.name, ln, f"class id {cls_id} out of range [0,{NC-1}]"))
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                bad_lines.append((txt_path.name, ln, f"coords out of [0,1]: {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}"))
            total_boxes += 1

    if empty_labels:
        print(f"  [WARN] {len(empty_labels)} empty label files (background images — OK if intentional)")
        for n in empty_labels[:5]:
            print(f"         {n}")

    if bad_lines:
        print(f"  [ERROR] {len(bad_lines)} malformed annotation lines:")
        for fname, ln, msg in bad_lines[:10]:
            print(f"         {fname}:{ln} — {msg}")
        if len(bad_lines) > 10:
            print(f"         ... and {len(bad_lines) - 10} more")
    else:
        print(f"  [OK] All annotation lines are valid ({total_boxes:,} boxes)")

    return {
        "split": split,
        "images": len(image_stems),
        "labels": len(label_stems),
        "missing_labels": len(missing_labels),
        "orphan_labels": len(orphan_labels),
        "empty_labels": len(empty_labels),
        "bad_lines": len(bad_lines),
        "total_boxes": total_boxes,
    }


def main():
    print("=" * 60)
    print("Dataset Integrity Validation")
    print(f"Root: {DATASET_ROOT}")
    print("=" * 60)

    stats = {}
    for split in ("train", "val"):
        stats[split] = validate_split(split)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_ok = True
    for split, s in stats.items():
        if not s:
            continue
        issues = s["missing_labels"] + s["orphan_labels"] + s["bad_lines"]
        status = "[OK]" if issues == 0 else f"[{issues} ISSUES]"
        print(f"  {split:5s}: {s['images']:6,} images | {s['total_boxes']:7,} boxes | {status}")
        if issues:
            all_ok = False

    if all_ok:
        print("\n  Dataset looks clean. Proceed to step 2.")
    else:
        print("\n  Fix the issues above before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Crop harvester for building a flour-sack classification dataset.

Runs the existing bag detector + bytetrack over a video and saves cropped bag
images to disk, ready to be sorted by hand into class folders (weight / brand /
sort).

Why this exists
---------------
The camera cannot be moved and the printed text on the sacks is unreadable, so
weight / brand / sort must be learned from each sack's *appearance* (colour band,
label layout, bag size). A visual classifier needs example crops of every product
variant. This script turns raw factory video into those crops with zero manual
frame-scrubbing.

Key design choice: DEDUPLICATION BY TRACK
-----------------------------------------
A single sack is visible for many frames, giving hundreds of near-identical
crops. That would make hand-sorting miserable. Instead we track each sack with
bytetrack and keep only the few SHARPEST crops per track (variance-of-Laplacian
focus score), so you get a small, diverse, mostly-in-focus set per bag.

Usage
-----
    # Harvest the current single product into a labelled folder:
    python harvest_crops.py --source dataset/flour_dataset/real_video/video.mp4 \
        --out crops/dikhanbaba_50kg_sort1

    # When you film a NEW product, harvest it into its own folder:
    python harvest_crops.py --source new_brand.mp4 --out crops/brandX_25kg_sort2

Then sort/verify the folders and point the (future) classifier trainer at them.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

import cv2
import heapq
from collections import defaultdict
from ultralytics import YOLO

DEFAULT_WEIGHTS = str(Path(__file__).resolve().parents[1] / "runs" / "bagdet_yolov8x2" / "weights" / "best.pt")


def sharpness(img) -> float:
    """Focus score: variance of the Laplacian. Higher = sharper."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def harvest(
    source: str,
    weights: str,
    out_dir: str,
    conf: float,
    iou: float,
    pad: float,
    min_side: int,
    per_track: int,
    every: int,
):
    os.makedirs(out_dir, exist_ok=True)
    model = YOLO(weights)

    # For each track we keep a min-heap of the best `per_track` crops seen so far,
    # keyed by sharpness so the weakest is evicted first.
    #   best[tid] = [(score, seq, crop_bgr), ...]  (heap by score)
    best: dict[int, list] = defaultdict(list)
    seq = 0  # tie-breaker so heap never compares numpy arrays

    results = model.track(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=640,
        device="0",
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        max_det=300,
        verbose=False,
    )

    frame_idx = 0
    for r in results:
        frame_idx += 1
        if every > 1 and (frame_idx % every) != 0:
            continue
        if r.boxes is None or r.boxes.id is None:
            continue

        frame = r.orig_img
        fh, fw = frame.shape[:2]
        boxes = r.boxes.xyxy.cpu().numpy()
        tids = r.boxes.id.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), tid in zip(boxes, tids):
            bw, bh = x2 - x1, y2 - y1
            # Padding around the box so the classifier sees full label context.
            px, py = bw * pad, bh * pad
            cx1 = max(0, int(x1 - px))
            cy1 = max(0, int(y1 - py))
            cx2 = min(fw, int(x2 + px))
            cy2 = min(fh, int(y2 + py))

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0 or min(crop.shape[:2]) < min_side:
                continue  # skip tiny / edge crops

            score = sharpness(crop)
            seq += 1
            h = best[tid]
            if len(h) < per_track:
                heapq.heappush(h, (score, seq, crop))
            elif score > h[0][0]:
                heapq.heapreplace(h, (score, seq, crop))

        if frame_idx % 200 == 0:
            print(f"frame {frame_idx:6d} | tracks so far: {len(best)}")

    # Flush the best crops to disk.
    stem = os.path.splitext(os.path.basename(source))[0]
    saved = 0
    for tid, h in best.items():
        for score, _, crop in sorted(h, reverse=True):
            fname = f"{stem}_track{tid:04d}_s{int(score):05d}.jpg"
            cv2.imwrite(os.path.join(out_dir, fname), crop)
            saved += 1

    print(f"\n{'='*50}")
    print(f"Frames processed : {frame_idx}")
    print(f"Unique bags       : {len(best)}")
    print(f"Crops saved       : {saved}  ->  {out_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Harvest bag crops for classification.")
    p.add_argument("--source", required=True, help="Input video path")
    p.add_argument("--out", required=True, help="Output folder for crops")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLO detector weights")
    p.add_argument("--conf", type=float, default=0.1, help="Detection confidence")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU")
    p.add_argument("--pad", type=float, default=0.08,
                   help="Fractional padding added around each box (0.08 = 8%%)")
    p.add_argument("--min-side", type=int, default=40,
                   help="Skip crops whose shorter side is below this many px")
    p.add_argument("--per-track", type=int, default=5,
                   help="How many sharpest crops to keep per tracked bag")
    p.add_argument("--every", type=int, default=1,
                   help="Process every Nth frame (speed vs. coverage)")
    args = p.parse_args()

    harvest(
        source=args.source,
        weights=args.weights,
        out_dir=args.out,
        conf=args.conf,
        iou=args.iou,
        pad=args.pad,
        min_side=args.min_side,
        per_track=args.per_track,
        every=args.every,
    )

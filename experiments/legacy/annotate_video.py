"""
Annotate a conveyor video: draw each bag's box with a fixed product label
(brand / weight / sort) plus a line-crossing counter, and save the result.

The label is hard-coded because this line runs a single product
(Dikhan Baba, 50 kg, Sort 1). Counting logic mirrors infer_video.py.

Usage:
    python annotate_video.py --source dataset/flour_dataset/real_video/video.mp4 \
        --save video_annotated.mp4
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

import cv2
from collections import defaultdict
from ultralytics import YOLO

DEFAULT_WEIGHTS = str(Path(__file__).resolve().parents[2] / "runs" / "bagdet_yolov8x2" / "weights" / "best.pt")

BRAND = "Dikhan Baba"
WEIGHT = "50 kg"
SORT = "Sort 1"


def draw_label(img, x1, y1, lines, color=(0, 200, 0)):
    """Draw a multi-line label box anchored at the top-left of a detection."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.5, 1
    pad = 4
    sizes = [cv2.getTextSize(t, font, scale, thick)[0] for t in lines]
    bw = max(w for w, _ in sizes) + 2 * pad
    lh = max(h for _, h in sizes) + 6
    bh = lh * len(lines) + pad

    # Keep the label on-screen (place above the box, or below if no room).
    ly = y1 - bh
    if ly < 0:
        ly = y1
    lx = min(max(0, x1), img.shape[1] - bw)

    cv2.rectangle(img, (lx, ly), (lx + bw, ly + bh), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        ty = ly + pad + lh * (i + 1) - 4
        cv2.putText(img, t, (lx + pad, ty), font, scale, color, thick, cv2.LINE_AA)


def run(source, weights, conf, iou, line_pos, min_hits, save):
    model = YOLO(weights)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    line_y = int(frame_h * line_pos)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save, fourcc, fps, (frame_w, frame_h))

    track_hits = defaultdict(int)
    track_min_cy = defaultdict(lambda: float("inf"))
    track_max_cy = defaultdict(lambda: float("-inf"))
    last_cy = {}
    counted_ids = set()
    bag_count = 0
    frame_idx = 0

    results = model.track(
        source=source, conf=conf, iou=iou, imgsz=640, device="0",
        stream=True, persist=True, tracker="bytetrack.yaml",
        max_det=300, verbose=False,
    )

    for r in results:
        frame_idx += 1
        frame = r.orig_img.copy()

        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            tids = r.boxes.id.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), tid, cf in zip(boxes, tids, confs):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cy = (y1 + y2) / 2.0

                track_hits[tid] += 1
                track_min_cy[tid] = min(track_min_cy[tid], cy)
                track_max_cy[tid] = max(track_max_cy[tid], cy)
                if tid in last_cy and track_hits[tid] >= min_hits and tid not in counted_ids:
                    if (last_cy[tid] < line_y <= cy) or (last_cy[tid] > line_y >= cy):
                        counted_ids.add(tid)
                        bag_count += 1
                last_cy[tid] = cy

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                draw_label(frame, x1, y1,
                           [f"{BRAND}", f"{WEIGHT}  {SORT}", f"id{tid} {cf:.2f}"])

        # counting line
        cv2.line(frame, (0, line_y), (frame_w, line_y), (0, 0, 255), 2)

        # counter overlay
        label = f"Bags counted: {bag_count}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (6, 6), (16 + tw, 16 + th + 6), (0, 0, 0), -1)
        cv2.putText(frame, label, (11, 11 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        writer.write(frame)
        if frame_idx % 200 == 0:
            print(f"frame {frame_idx:6d} | counted={bag_count}")

    # interpolation pass for bags whose detection dropped at the line
    for tid in track_hits:
        if tid not in counted_ids and track_hits[tid] >= min_hits:
            if track_min_cy[tid] < line_y < track_max_cy[tid]:
                counted_ids.add(tid)
                bag_count += 1

    writer.release()
    print(f"\n{'='*50}\nFrames: {frame_idx} | Total bags counted: {bag_count}")
    print(f"Saved: {save}\n{'='*50}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--save", required=True)
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--line", type=float, default=0.5, help="counting line position (0-1 of height)")
    p.add_argument("--min-hits", type=int, default=3)
    a = p.parse_args()
    run(a.source, a.weights, a.conf, a.iou, a.line, a.min_hits, a.save)

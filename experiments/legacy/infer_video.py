"""
Inference script for bag counting on conveyor belt video.

Counting method: line crossing with bytetrack + end-of-track interpolation.
- bytetrack uses spatial proximity only (IoU), so identical-looking bags are
  never merged into one track.
- A bag is counted when its centre crosses the counting line.
- If detection dropped out at the exact crossing moment, a post-track check
  catches it: any track whose first and last cx values span the line is counted.

Usage:
    python infer_video.py --source test_video.mp4 --save output.mp4
    python infer_video.py --source test_video.mp4 --line 0.5 --save output.mp4
    python infer_video.py --source test_video.mp4 --show
"""

import argparse
import os
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

import cv2
from collections import defaultdict
from ultralytics import YOLO


def run_inference(
    source: str,
    weights: str,
    conf: float,
    iou: float,
    line_pos: float,
    min_hits: int,
    show: bool,
    save: str | None,
):
    model = YOLO(weights)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    # Horizontal counting line (bags travel top-to-bottom)
    line_y = int(frame_h * line_pos)

    writer = None
    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save, fourcc, fps, (frame_w, frame_h))

    # Tracking state
    track_hits:   dict[int, int]         = defaultdict(int)   # tid -> frames seen
    track_min_cy: dict[int, float]       = defaultdict(lambda: float('inf'))
    track_max_cy: dict[int, float]       = defaultdict(lambda: float('-inf'))
    last_cy:      dict[int, float]       = {}                  # tid -> most recent cy
    counted_ids:  set[int]               = set()
    bag_count = 0
    frame_idx = 0

    results = model.track(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=640,
        device="0",
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",  # spatial-only matching → no ID merging across locations
        agnostic_nms=False,
        max_det=300,
        augment=True,              # TTA for more stable boxes
        verbose=False,
    )

    for r in results:
        frame_idx += 1
        annotated = r.plot()

        if r.boxes is not None and r.boxes.id is not None:
            boxes     = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy().astype(int)

            for box, tid in zip(boxes, track_ids):
                cy = (box[1] + box[3]) / 2.0

                track_hits[tid] += 1
                track_min_cy[tid] = min(track_min_cy[tid], cy)
                track_max_cy[tid] = max(track_max_cy[tid], cy)

                # Real-time line crossing check
                if tid in last_cy and track_hits[tid] >= min_hits and tid not in counted_ids:
                    crossed = (last_cy[tid] < line_y <= cy) or (last_cy[tid] > line_y >= cy)
                    if crossed:
                        counted_ids.add(tid)
                        bag_count += 1

                last_cy[tid] = cy

        # --- draw counting line ---
        cv2.line(annotated, (0, line_y), (frame_w, line_y), (0, 0, 255), 3)

        # --- overlay count ---
        label = f"Bags counted: {bag_count}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(annotated, (8, 8), (18 + tw, 18 + th + 8), (0, 0, 0), -1)
        cv2.putText(annotated, label, (13, 13 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if writer:
            writer.write(annotated)

        if show:
            cv2.imshow("Bag Counter", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        print(f"Frame {frame_idx:05d} | counted={bag_count:4d} | active_tracks={len(track_hits):3d}")

    # --- Post-track interpolation pass ---
    # Catches bags where detection dropped out mid-journey.
    # Uses min/max cy across ALL frames of the track (not just first/last),
    # so a bag that was seen above the line and below the line at any point
    # is counted even if detection vanished exactly at the crossing moment.
    late_counts = 0
    for tid in track_hits:
        if tid not in counted_ids and track_hits[tid] >= min_hits:
            if track_min_cy[tid] < line_y < track_max_cy[tid]:
                counted_ids.add(tid)
                bag_count += 1
                late_counts += 1

    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print(f"Total frames processed  : {frame_idx}")
    print(f"Unique bags counted     : {bag_count}")
    if late_counts:
        print(f"  (of which {late_counts} caught by interpolation pass)")
    if save:
        print(f"Annotated video saved   : {save}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count flour bags on conveyor belt.")
    parser.add_argument("--source",   required=True,
                        help="Path to video file (or 0 for webcam)")
    parser.add_argument("--weights",
                        default=str(Path(__file__).resolve().parents[2] / "runs" / "bagdet_yolov8x2" / "weights" / "best.pt"),
                        help="Path to YOLO weights (.pt)")
    parser.add_argument("--conf",     type=float, default=0.1,
                        help="Detection confidence threshold")
    parser.add_argument("--iou",      type=float, default=0.5,
                        help="NMS IoU threshold")
    parser.add_argument("--line",     type=float, default=0.6,
                        help="Counting line position as fraction of frame height (0-1). "
                             "Default 0.6 = a bit below centre. 0=top, 1=bottom.")
    parser.add_argument("--min_hits", type=int,   default=3,
                        help="Min frames a track must appear before being counted (filters noise)")
    parser.add_argument("--show",     action="store_true",
                        help="Display annotated video in a window (requires a display)")
    parser.add_argument("--save",     default=None, metavar="OUTPUT.mp4",
                        help="Save annotated video to this file")
    args = parser.parse_args()

    run_inference(
        source=args.source,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        line_pos=args.line,
        min_hits=args.min_hits,
        show=args.show,
        save=args.save,
    )

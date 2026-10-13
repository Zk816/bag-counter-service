"""
YOLOv8x training for flour bag detection on conveyor belt.
Optimized for high recall (minimize missed bags).
Hardware: 8x NVIDIA H200
"""

import os
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

model = YOLO("yolov8x.pt")

results = model.train(
    data=str(ROOT / "dataset" / "flour_dataset" / "data.yaml"),
    epochs=100,
    imgsz=640,
    batch=32,                   # 16 per GPU across 2x H200
    device="0",                 # first GPU in CUDA_VISIBLE_DEVICES
    workers=8,                  # data loading workers
    project=str(ROOT / "runs"),
    name="bagdet_yolov8x",
    exist_ok=False,

    # --- Optimizer ---
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,                   # final lr = lr0 * lrf
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,

    # --- Recall-focused loss weights ---
    # Lowering cls raises recall by making the classifier less conservative
    box=7.5,
    cls=0.3,                    # default 0.5 → lower to bias toward recall
    dfl=1.5,

    # --- Augmentation (strong, for factory lighting variation) ---
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,                # slight rotation (bags on belt are roughly upright)
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.0005,
    flipud=0.0,                 # bags on belt don't flip vertically
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,

    # --- Recall-oriented NMS & thresholds ---
    conf=0.001,                 # very low conf threshold during val for proper mAP curve
    iou=0.7,                    # NMS IoU; higher = keep more overlapping boxes (dense belts)
    max_det=300,                # max detections per frame; raise if belt is very dense

    # --- Misc ---
    save=True,
    save_period=10,             # checkpoint every 10 epochs
    patience=30,                # early stop if no improvement for 30 epochs
    seed=42,
    amp=True,                   # mixed precision for speed
    plots=True,
    verbose=True,
)

print(f"\nBest weights saved to: {results.save_dir}/weights/best.pt")

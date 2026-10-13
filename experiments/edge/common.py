"""Paths and settings shared by the edge experiments (CPU deployment of the bag counter)."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EDGE = ROOT / "experiments" / "edge"
RESULTS = EDGE / "results"
RESULTS.mkdir(exist_ok=True)
DATA_YAML = ROOT / "dataset" / "flour_dataset" / "data.yaml"
VAL_YAML = EDGE / "val1000.yaml"           # fixed 1000-image val subset: CPU evaluation in minutes, not hours
VIDEOS = [ROOT / "test_video.mp4", ROOT / "test_video2.mp4", ROOT / "test_video3.mp4"]

WEIGHTS = {
    "yolov8x": ROOT / "runs" / "bagdet_yolov8x2" / "weights" / "best.pt",     # the original model (68M params)
    "yolov8s": ROOT / "runs" / "edge" / "v8s" / "weights" / "best.pt",
    "yolov8n": ROOT / "runs" / "edge" / "v8n" / "weights" / "best.pt",
}
IMGSZ = 640
CONF, IOU, LINE, MIN_HITS = 0.1, 0.5, 0.6, 3          # same counting settings as infer_video.py

# An edge box has a few cores, not 224: every CPU number is measured pinned to 8 cores.
CPU_CORES = list(range(int(os.environ.get("EDGE_CORE0", "208")), int(os.environ.get("EDGE_CORE0", "208")) + 8))
THREADS = len(CPU_CORES)


def pin_cpu() -> None:
    os.sched_setaffinity(0, CPU_CORES)


def ov_dir(model: str, fmt: str) -> Path:
    """Exported OpenVINO folder: fmt 'fp32' or 'int8'."""
    w = WEIGHTS[model]
    return w.parent / (f"{w.stem}_int8_openvino_model" if fmt == "int8" else f"{w.stem}_openvino_model")

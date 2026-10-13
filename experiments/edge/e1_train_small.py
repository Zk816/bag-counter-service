"""Train the small detectors (YOLOv8n, YOLOv8s) on the same 27.7k frames as the original YOLOv8x.

The x model is accurate but 258 GFLOPs per frame: on 8 CPU cores it cannot keep up with one camera.
The question for the edge box is how much accuracy n (8.7 GFLOPs) and s (28.6) give up.
"""

import torch
from ultralytics import YOLO

from common import DATA_YAML, ROOT, WEIGHTS

EPOCHS = 60


def finished(name: str) -> bool:
    """Ultralytics strips the optimizer from last.pt only when training completes."""
    last = WEIGHTS[name].with_name("last.pt")
    if not last.exists():
        return False
    return torch.load(last, map_location="cpu", weights_only=False).get("optimizer") is None


def main() -> None:
    for name in ("yolov8n", "yolov8s"):
        if finished(name):
            print(f"skip {name}: training already finished")
            continue
        last = WEIGHTS[name].with_name("last.pt")
        if last.exists():                       # interrupted run: continue where it stopped
            YOLO(str(last)).train(resume=True)
            continue
        YOLO(f"{name}.pt").train(data=str(DATA_YAML), epochs=EPOCHS, imgsz=640, batch=64, workers=16, device=0,
                                 project=str(ROOT / "runs" / "edge"), name=name.replace("yolo", ""), exist_ok=True,
                                 patience=15, cos_lr=True, close_mosaic=10, seed=0, plots=True)


if __name__ == "__main__":
    main()

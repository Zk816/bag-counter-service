"""Export every detector to ONNX, OpenVINO FP32 and OpenVINO INT8 (NNCF post-training quantisation).

INT8 calibration uses 300 training frames; accuracy is then checked on the val subset in e3, so
any loss from quantisation is measured, not assumed.
"""

import random
import shutil

from ultralytics import YOLO

from common import DATA_YAML, IMGSZ, ROOT, VAL_YAML, WEIGHTS, ov_dir


def make_val_subset() -> None:
    """1000 fixed val images, as a file list Ultralytics accepts in place of a folder."""
    if VAL_YAML.exists():
        return
    images = sorted((ROOT / "dataset" / "flour_dataset" / "images" / "val").glob("*.jpg"))
    pick = sorted(random.Random(0).sample(images, 1000))
    lst = VAL_YAML.with_suffix(".txt")
    lst.write_text("\n".join(str(p) for p in pick) + "\n")
    VAL_YAML.write_text(f"path: {ROOT / 'dataset' / 'flour_dataset'}\ntrain: {lst}\nval: {lst}\nnc: 1\nnames: [bag]\n")


def stale(path, weights) -> bool:
    """Missing, or older than the weights (the model was retrained since)."""
    return not path.exists() or path.stat().st_mtime < weights.stat().st_mtime


def main() -> None:
    make_val_subset()
    for name, w in WEIGHTS.items():
        if not w.exists():
            print(f"!! {name}: no weights at {w}, skipped")
            continue
        model = YOLO(str(w))
        if stale(w.with_suffix(".onnx"), w):
            model.export(format="onnx", imgsz=IMGSZ, opset=17, simplify=True)
        if stale(ov_dir(name, "fp32"), w):
            shutil.rmtree(ov_dir(name, "fp32"), ignore_errors=True)
            model.export(format="openvino", imgsz=IMGSZ, half=False)
        if stale(ov_dir(name, "int8"), w):
            shutil.rmtree(ov_dir(name, "int8"), ignore_errors=True)
            model.export(format="openvino", imgsz=IMGSZ, int8=True, data=str(DATA_YAML), split="train", fraction=0.011)
        print(f"{name}: exported")


if __name__ == "__main__":
    main()

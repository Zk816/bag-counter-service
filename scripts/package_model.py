from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2

from bagcounter.detect.openvino import Detector, sha256

SELFTEST_MIN_SCORE = 0.5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ir", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--selftest-video", type=Path, required=True)
    p.add_argument("--selftest-at", type=float, default=10.0, help="seconds into the video")
    p.add_argument("--metrics", default="{}")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    xml = next(a.ir.glob("*.xml"))
    shutil.copy(xml, a.out / "model.xml")
    shutil.copy(xml.with_suffix(".bin"), a.out / "model.bin")

    cap = cv2.VideoCapture(str(a.selftest_video))
    cap.set(cv2.CAP_PROP_POS_MSEC, a.selftest_at * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read a frame at {a.selftest_at}s from {a.selftest_video}")
    cv2.imwrite(str(a.out / "selftest.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    frame = cv2.imread(str(a.out / "selftest.jpg"))

    meta = {"name": a.name, "version": a.version, "imgsz": 640, "classes": ["bag"],
            "source": str(a.ir), "metrics": json.loads(a.metrics), "selftest": {"image": "selftest.jpg"}}
    files = ["model.xml", "model.bin", "selftest.jpg"]
    (a.out / "model.json").write_text(json.dumps(meta, indent=2))
    (a.out / "checksums.sha256").write_text("".join(f"{sha256(a.out / f)}  {f}\n" for f in [*files, "model.json"]))

    det = Detector(a.out, threads=8, conf=0.1, iou=0.5, max_det=300, hint="LATENCY")
    boxes = det.session().detect(frame)
    boxes = boxes[boxes[:, 4] >= SELFTEST_MIN_SCORE]
    if len(boxes) == 0:
        raise SystemExit("self-test frame has no confident detection: pick another --selftest-at")
    meta["selftest"]["min_score"] = SELFTEST_MIN_SCORE
    meta["selftest"]["boxes"] = [[round(float(v), 2) for v in b[:5]] for b in boxes]
    (a.out / "model.json").write_text(json.dumps(meta, indent=2))
    (a.out / "checksums.sha256").write_text("".join(f"{sha256(a.out / f)}  {f}\n" for f in [*files, "model.json"]))
    print(f"packaged {a.name}:{a.version} -> {a.out} ({len(boxes)} self-test boxes)")


if __name__ == "__main__":
    main()

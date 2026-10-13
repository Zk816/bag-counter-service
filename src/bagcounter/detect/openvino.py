from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import openvino as ov
from openvino.preprocess import PrePostProcessor

from bagcounter.errors import ModelError

Boxes = npt.NDArray[np.float32]
Frame = npt.NDArray[Any]


@dataclass(frozen=True)
class ModelInfo:
    name: str
    version: str
    imgsz: int
    classes: list[str]
    meta: dict[str, Any]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_model_dir(model_dir: Path) -> ModelInfo:
    sums = model_dir / "checksums.sha256"
    meta_file = model_dir / "model.json"
    if not sums.is_file() or not meta_file.is_file():
        raise ModelError("model directory is incomplete", context={"model_dir": str(model_dir)})
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        f = model_dir / name.strip().lstrip("*")
        if not f.is_file():
            raise ModelError("model file missing", context={"file": f.name})
        if sha256(f) != digest:
            raise ModelError("model checksum mismatch", context={"file": f.name})
    try:
        meta = json.loads(meta_file.read_text())
        return ModelInfo(meta["name"], meta["version"], int(meta["imgsz"]), list(meta["classes"]), meta)
    except (ValueError, KeyError, TypeError) as e:
        raise ModelError("model.json is invalid") from e


def letterbox(frame: Frame, size: int) -> tuple[Frame, float, tuple[int, int]]:
    h, w = frame.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, r, (left, top)


def decode(out: npt.NDArray[np.float32], r: float, pad: tuple[int, int], shape: tuple[int, int],
           conf: float, iou: float, max_det: int) -> Boxes:
    p = out[0].T
    cls_scores = p[:, 4:]
    cls = cls_scores.argmax(1)
    score = cls_scores[np.arange(len(p)), cls]
    keep = score >= conf
    p, cls, score = p[keep], cls[keep], score[keep]
    if not len(p):
        return np.zeros((0, 6), np.float32)
    cx, cy = (p[:, 0] - pad[0]) / r, (p[:, 1] - pad[1]) / r
    w, h = p[:, 2] / r, p[:, 3] / r
    xywh = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)
    nms = cv2.dnn.NMSBoxesBatched(xywh.tolist(), score.tolist(), cls.tolist(), conf, iou)
    idx = np.asarray(nms, dtype=np.int64).reshape(-1)[:max_det]
    b = xywh[idx]
    x1 = np.clip(b[:, 0], 0, shape[1])
    y1 = np.clip(b[:, 1], 0, shape[0])
    x2 = np.clip(b[:, 0] + b[:, 2], 0, shape[1])
    y2 = np.clip(b[:, 1] + b[:, 3], 0, shape[0])
    return np.stack([x1, y1, x2, y2, score[idx], cls[idx]], axis=1).astype(np.float32)


class Detector:

    def __init__(self, model_dir: Path, *, threads: int, conf: float, iou: float, max_det: int,
                 hint: str = "LATENCY") -> None:
        self.info = verify_model_dir(model_dir)
        self.conf, self.iou, self.max_det = conf, iou, max_det
        try:
            core = ov.Core()
            model = core.read_model(next(model_dir.glob("*.xml")))
            ppp = PrePostProcessor(model)
            ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC")).set_color_format(
                ov.preprocess.ColorFormat.BGR)
            ppp.input().model().set_layout(ov.Layout("NCHW"))
            ppp.input().preprocess().convert_color(ov.preprocess.ColorFormat.RGB).convert_element_type(
                ov.Type.f32).scale(255.0)
            self._compiled = core.compile_model(ppp.build(), "CPU", {"INFERENCE_NUM_THREADS": threads,
                                                                     "PERFORMANCE_HINT": hint})
        except (RuntimeError, StopIteration) as e:
            raise ModelError("OpenVINO could not load the model", context={"model_dir": str(model_dir)}) from e

    @property
    def version(self) -> str:
        return f"{self.info.name}:{self.info.version}"

    def session(self) -> DetectorSession:
        return DetectorSession(self)


class DetectorSession:

    def __init__(self, det: Detector) -> None:
        self._det = det
        self._req = det._compiled.create_infer_request()

    def detect(self, frame: Frame) -> Boxes:
        d = self._det
        canvas, r, pad = letterbox(frame, d.info.imgsz)
        try:
            out = self._req.infer({0: canvas[None]})[0]
        except RuntimeError as e:
            raise ModelError("inference failed") from e
        return decode(out, r, pad, (frame.shape[0], frame.shape[1]), d.conf, d.iou, d.max_det)

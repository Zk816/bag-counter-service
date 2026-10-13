from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from bagcounter.config import Settings
from bagcounter.detect.openvino import Detector, verify_model_dir
from bagcounter.errors import AppError, ModelError
from bagcounter.media import tools_available

log = logging.getLogger(__name__)

SELFTEST_IOU = 0.9
SELFTEST_SCORE_TOL = 0.05


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    ms: float


@dataclass
class PreflightResult:
    checks: list[CheckResult] = field(default_factory=list)
    detector: Detector | None = None

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def selftest(det: Detector, settings: Settings) -> str:
    spec = det.info.meta.get("selftest") or {}
    img = cv2.imread(str(settings.model_dir / str(spec.get("image", ""))))
    if img is None or "boxes" not in spec:
        raise ModelError("model package has no self-test data")
    got = det.session().detect(img)
    got = got[got[:, 4] >= float(spec["min_score"]) - SELFTEST_SCORE_TOL]
    want: list[list[float]] = spec["boxes"]
    unmatched = []
    for w in want:
        ious = [_iou(w, [float(v) for v in g[:4]]) for g in got]
        j = int(np.argmax(ious)) if ious else -1
        if j < 0 or ious[j] < SELFTEST_IOU or abs(float(got[j][4]) - w[4]) > SELFTEST_SCORE_TOL:
            unmatched.append(w)
    if unmatched or len(got[got[:, 4] >= float(spec["min_score"]) + SELFTEST_SCORE_TOL]) > len(want):
        raise ModelError("model self-test output differs from the packaged expectation",
                         context={"expected": len(want), "got": len(got), "unmatched": len(unmatched)})
    return f"{len(want)} boxes match"


def run_preflight(settings: Settings, *, load_model: bool = True) -> PreflightResult:
    res = PreflightResult()

    def check(name: str, fn: Callable[[], str]) -> bool:
        t = time.perf_counter()
        try:
            detail, ok = fn(), True
        except (AppError, OSError) as e:
            detail, ok = f"{type(e).__name__}: {e}", False
        res.checks.append(CheckResult(name, ok, detail, round(1000 * (time.perf_counter() - t), 1)))
        log.log(logging.INFO if ok else logging.CRITICAL, "preflight check", extra={"check": name, "ok": ok,
                                                                                    "detail": detail})
        return ok

    def data_dir() -> str:
        for d in (settings.data_dir, settings.uploads_dir, settings.outputs_dir):
            d.mkdir(parents=True, exist_ok=True)
        probe = settings.data_dir / ".write_probe"
        probe.write_bytes(b"ok")
        probe.unlink()
        free = shutil.disk_usage(settings.data_dir).free
        if free < settings.min_free_disk_bytes:
            raise OSError(f"only {free} bytes free under {settings.data_dir}")
        return f"writable, {free // 2**20} MiB free"

    def tools() -> str:
        missing = [k for k, v in tools_available().items() if not v]
        if missing:
            raise OSError(f"missing: {', '.join(missing)}")
        return "ffmpeg, ffprobe"

    def model_files() -> str:
        info = verify_model_dir(settings.model_dir)
        return f"{info.name}:{info.version} checksums ok"

    def model_load() -> str:
        res.detector = Detector(settings.model_dir, threads=settings.infer_threads, conf=settings.conf_threshold,
                                iou=settings.nms_iou, max_det=settings.max_detections,
                                hint=settings.ov_performance_hint)
        return "compiled on CPU"

    def model_selftest() -> str:
        if res.detector is None:
            raise ModelError("model not loaded")
        return selftest(res.detector, settings)

    ok = check("data_dir", data_dir) & check("ffmpeg", tools) & check("model_files", model_files)
    if load_model and ok:
        _ = check("model_load", model_load) and check("model_selftest", model_selftest)
    return res

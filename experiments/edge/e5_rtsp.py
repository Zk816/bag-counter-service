"""Live RTSP test: N cameras -> one 8-core box, real-time decode, detection, tracking, counting.

Setup: a local RTSP server (MediaMTX in Docker, 127.0.0.1:8564) and one ffmpeg publisher per
camera, each looping a conveyor video in real time (-re, H.264 passthrough, ~30 FPS).

Per camera, a reader thread decodes the RTSP stream and keeps only the NEWEST frame (a queue of 1):
when inference is slower than the camera, old frames are dropped instead of piling up, so latency
stays bounded — the standard choice for live analytics (a growing buffer means minutes-old counts).
One inference loop runs OpenVINO in THROUGHPUT mode with one async request per camera.

Measured per configuration (model x number of cameras), over DURATION seconds:
  processed FPS per camera, share of camera frames dropped, end-to-end latency
  (frame decoded -> count updated) p50/p95, and whether all cameras agree on the count.
"""

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

from common import RESULTS, ROOT, ov_dir, pin_cpu
from pipeline import Counter, compile_model, decode, letterbox

RTSP = "rtsp://127.0.0.1:8564"        # not 8554, to avoid clashing with an RTSP server already on the host
DURATION = 45
VIDEO = ROOT / "test_video2.mp4"          # the 60 s one: no loop restart inside a run
CONTAINER = "bagcounter-rtsp-bench"


def start_server() -> None:
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True, timeout=30, check=False)
    subprocess.run(["docker", "run", "-d", "--name", CONTAINER, "-p", "127.0.0.1:8564:8554",
                    "bluenviron/mediamtx:latest"], check=True, capture_output=True, timeout=300)
    time.sleep(3)


def publish(cams: int) -> list[subprocess.Popen]:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    procs = [subprocess.Popen([ffmpeg, "-loglevel", "error", "-re", "-stream_loop", "-1", "-i", str(VIDEO),
                               "-an", "-c:v", "copy", "-f", "rtsp", "-rtsp_transport", "tcp", f"{RTSP}/cam{i}"])
             for i in range(cams)]
    time.sleep(3)
    return procs


class Reader(threading.Thread):
    """Decodes one RTSP stream; holds only the latest frame and counts what it had to drop."""

    def __init__(self, url: str) -> None:
        super().__init__(daemon=True)
        self.url, self.lock = url, threading.Lock()
        self.frame: tuple[np.ndarray, float] | None = None
        self.decoded = self.dropped = 0
        self.stop = False

    def run(self) -> None:
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while not self.stop:
            ok, f = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                if self.frame is not None:
                    self.dropped += 1
                self.frame = (f, time.perf_counter())
                self.decoded += 1
        cap.release()

    def take(self) -> tuple[np.ndarray, float] | None:
        with self.lock:
            f, self.frame = self.frame, None
        return f


def run_config(model_dir: Path, cams: int) -> dict:
    readers = [Reader(f"{RTSP}/cam{i}") for i in range(cams)]
    for r in readers:
        r.start()
    time.sleep(3)
    compiled = compile_model(model_dir, hint="THROUGHPUT")
    reqs = [compiled.create_infer_request() for _ in range(cams)]
    counters: list[Counter | None] = [None] * cams
    processed = [0] * cams
    latencies: list[float] = []
    for r in readers:            # start measuring from here
        r.decoded = r.dropped = 0
    start = time.perf_counter()
    while time.perf_counter() - start < DURATION:
        jobs = []
        for i, (r, req) in enumerate(zip(readers, reqs)):
            item = r.take()
            if item is None:
                continue
            frame, t0 = item
            x, scale, pad = letterbox(frame)
            req.start_async({0: x})
            jobs.append((i, req, frame, t0, scale, pad))
        if not jobs:
            time.sleep(0.002)
            continue
        for i, req, frame, t0, scale, pad in jobs:
            req.wait()
            if counters[i] is None:
                counters[i] = Counter(frame.shape[0], 30.0)
            counters[i].update(decode(req.get_output_tensor(0).data, scale, pad, frame.shape[:2]), frame.shape[:2])
            processed[i] += 1
            latencies.append((time.perf_counter() - t0) * 1000)
    elapsed = time.perf_counter() - start
    for r in readers:
        r.stop = True
    counts = [len(c.counted) if c else 0 for c in counters]
    decoded = sum(r.decoded for r in readers)
    return {"cameras": cams, "fps_per_camera": [p / elapsed for p in processed],
            "camera_fps": decoded / cams / elapsed,
            "dropped_share": sum(r.dropped for r in readers) / max(decoded, 1),
            "latency_ms_p50": float(np.percentile(latencies, 50)) if latencies else None,
            "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else None,
            "counts": counts}


def main() -> None:
    pin_cpu()
    start_server()
    rows = []
    try:
        for model in ("yolov8n", "yolov8s", "yolov8x"):
            d = ov_dir(model, "int8")
            if not d.exists():
                continue
            for cams in (1, 2, 4):
                pubs = publish(cams)
                try:
                    row = {"model": model, "format": "int8", **run_config(d, cams)}
                finally:
                    for p in pubs:
                        p.terminate()
                        p.wait(timeout=10)
                rows.append(row)
                print(row, flush=True)
                (RESULTS / "rtsp.json").write_text(json.dumps({"duration_s": DURATION, "rows": rows}, indent=2))
                time.sleep(2)
    finally:
        subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True, timeout=30, check=False)


if __name__ == "__main__":
    main()

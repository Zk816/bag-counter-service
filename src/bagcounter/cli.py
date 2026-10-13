from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bagcounter.config import get_settings


def _count(a: argparse.Namespace) -> int:
    from bagcounter.detect.openvino import Detector
    from bagcounter.geometry import Line
    from bagcounter.jobs import run_video
    from bagcounter.video.sources import FileSource

    s = get_settings()
    line = Line.parse(a.line, a.direction)
    det = Detector(s.model_dir, threads=s.infer_threads, conf=s.conf_threshold, iou=s.nms_iou,
                   max_det=s.max_detections, hint="LATENCY")
    src = FileSource(Path(a.video), max_seconds=s.max_video_seconds, max_pixels=s.max_video_pixels, stride=a.stride)
    summary = run_video(src, line, s, det, Path(a.out), annotate=not a.no_video)
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return 0


def _preflight(_: argparse.Namespace) -> int:
    from bagcounter.preflight import run_preflight

    res = run_preflight(get_settings())
    for c in res.checks:
        sys.stdout.write(f"{'PASS' if c.ok else 'FAIL'}  {c.name:<15} {c.ms:>8.1f} ms  {c.detail}\n")
    sys.stdout.write(("ready to serve" if res.ok else "NOT ready: the worker would refuse to start") + "\n")
    return 0 if res.ok else 1


def _healthcheck(a: argparse.Namespace) -> int:
    import time
    import urllib.request

    from bagcounter.errors import StorageError
    from bagcounter.store import Store

    s = get_settings()
    if a.component == "api":
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{s.port}/healthz", timeout=3) as r:
                return 0 if r.status == 200 else 1
        except OSError:
            return 1
    try:
        hb = Store(s.db_path).get_meta("worker_heartbeat")
    except StorageError:
        return 1
    return 0 if hb and time.time() - hb[1] < s.heartbeat_stale_s else 1


def _api(_: argparse.Namespace) -> int:
    import uvicorn

    s = get_settings()
    uvicorn.run("bagcounter.api.app:create_app", factory=True, host=s.host, port=s.port, log_config=None,
                timeout_graceful_shutdown=20)
    return 0


def _worker(_: argparse.Namespace) -> int:
    from bagcounter.worker import main

    return int(main())


def main() -> None:
    p = argparse.ArgumentParser(prog="bagcounter")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("count", help="count bags in a video file")
    c.add_argument("video")
    c.add_argument("--line", required=True, help="ax,ay,bx,by normalised to 0..1")
    c.add_argument("--direction", default="positive", choices=["positive", "negative", "both"])
    c.add_argument("--out", required=True)
    c.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    c.add_argument("--no-video", action="store_true", help="skip the annotated video")
    c.set_defaults(fn=_count)
    sub.add_parser("preflight", help="verify model, runtime and storage; exit 1 on failure").set_defaults(fn=_preflight)
    sub.add_parser("api", help="run the REST API + web UI").set_defaults(fn=_api)
    h = sub.add_parser("healthcheck", help="container health probe; exit 0 when healthy")
    h.add_argument("component", choices=["api", "worker"])
    h.set_defaults(fn=_healthcheck)
    sub.add_parser("worker", help="run the video job + camera worker").set_defaults(fn=_worker)
    a = p.parse_args()
    raise SystemExit(a.fn(a))


if __name__ == "__main__":
    main()

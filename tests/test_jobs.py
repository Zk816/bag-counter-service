import json
import sys
from pathlib import Path

import pytest

from bagcounter.detect.openvino import Detector
from bagcounter.geometry import Line
from bagcounter.jobs import run_video
from bagcounter.video.sources import FileSource

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from eval_count import evaluate

DATA = Path(__file__).parent / "data"
pytestmark = pytest.mark.integration


def test_clip_count_matches_ground_truth(settings, tmp_path):
    gt = json.loads((DATA / "clip_8s.gt.json").read_text())
    ln = gt["line"]
    line = Line.parse(f"{ln['ax']},{ln['ay']},{ln['bx']},{ln['by']}", ln["direction"])
    det = Detector(settings.model_dir, threads=4, conf=settings.conf_threshold, iou=settings.nms_iou,
                   max_det=settings.max_detections, hint="LATENCY")
    src = FileSource(DATA / gt["video"], max_seconds=60, max_pixels=settings.max_video_pixels)
    summary = run_video(src, line, settings, det, tmp_path / "out", annotate=True)

    report = json.loads((tmp_path / "out" / "report.json").read_text())
    score = evaluate(gt, report, tol=0.5)
    assert summary["count"] == gt["count"], score
    assert score["precision"] >= 0.88 and score["recall"] >= 0.88, score
    assert (tmp_path / "out" / "annotated.mp4").stat().st_size > 10_000
    assert len(list((tmp_path / "out" / "snapshots").glob("*.jpg"))) == summary["count"]

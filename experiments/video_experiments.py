from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_count import evaluate  # noqa: E402

from bagcounter.config import Settings  # noqa: E402
from bagcounter.detect.openvino import Detector  # noqa: E402
from bagcounter.geometry import line_from_dict  # noqa: E402
from bagcounter.jobs import run_video  # noqa: E402
from bagcounter.video.sources import FileSource  # noqa: E402

CORES = list(range(int(os.environ.get("EXP_CORE0", "208")), int(os.environ.get("EXP_CORE0", "208")) + 8))
OUT = ROOT / "experiments" / "results"
WORK = Path(os.environ.get("EXP_WORK", "/tmp/bagcounter_exp"))
CLIP = ROOT / "tests" / "data" / "clip_8s.mp4"
GT = json.loads((ROOT / "tests" / "data" / "clip_8s.gt.json").read_text())
FACTORY = ROOT / "dataset" / "flour_dataset" / "real_video" / "video.mp4"
FACTORY_LINE = {"ax": 0.05, "ay": 0.6, "bx": 0.95, "by": 0.6, "direction": "negative"}
BITRATES_K = [4000, 1000, 500, 250, 120]
FACTORY_BITRATES_K = [2000, 500, 200]
STRIDES = [1, 2, 3, 4, 6]


def encode(src: Path, codec: str, kbps: int, dst: Path) -> float:
    lib = {"h264": "libx264", "h265": "libx265"}[codec]
    extra = ["-x265-params", "log-level=error"] if codec == "h265" else []
    subprocess.run(["taskset", "-c", ",".join(map(str, CORES)), "ffmpeg", "-loglevel", "error", "-y", "-i", str(src),
                    "-an", "-c:v", lib, *extra, "-b:v", f"{kbps}k", "-maxrate", f"{kbps}k", "-bufsize", f"{2 * kbps}k",
                    "-preset", "medium", str(dst)], check=True, timeout=3600)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=bit_rate", "-of", "csv=p=0", str(dst)],
                           capture_output=True, text=True, check=True, timeout=60)
    return round(int(probe.stdout.strip()) / 1000)


def count(det: Detector, s: Settings, video: Path, line: dict[str, object], stride: int = 1) -> dict[str, object]:
    src = FileSource(video, max_seconds=3600, max_pixels=s.max_video_pixels, stride=stride)
    return run_video(src, line_from_dict(line), s, det, WORK / "out" / video.stem, annotate=False)


def score(summary: dict[str, object], video: Path) -> dict[str, object]:
    report = json.loads((WORK / "out" / video.stem / "report.json").read_text())
    return evaluate(GT, report, tol=0.5)


def decode_ms_per_frame(video: Path) -> float:
    out = subprocess.run(["taskset", "-c", str(CORES[0]), "ffmpeg", "-benchmark", "-threads", "1", "-i", str(video),
                          "-f", "null", "-"], capture_output=True, text=True, check=True, timeout=600)
    utime = float(re.search(r"utime=([\d.]+)s", out.stderr).group(1))  # type: ignore[union-attr]
    frames = int(re.findall(r"frame=\s*(\d+)", out.stderr)[-1])
    return round(1000 * utime / frames, 2)


def main() -> None:
    os.sched_setaffinity(0, CORES)
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    s = Settings(data_dir=WORK / "data", infer_threads=len(CORES))
    det = Detector(s.model_dir, threads=len(CORES), conf=s.conf_threshold, iou=s.nms_iou, max_det=s.max_detections)
    res: dict[str, list[dict[str, object]]] = {"codec_bitrate": [], "fps": [], "decode": [], "factory": []}

    gt_line = GT["line"]
    for codec in ("h264", "h265"):
        for kbps in BITRATES_K:
            v = WORK / f"clip_{codec}_{kbps}k.mp4"
            real = encode(CLIP, codec, kbps, v)
            sm = count(det, s, v, gt_line)
            row = {"codec": codec, "target_kbps": kbps, "kbps": real, **score(sm, v), "tracks": sm["tracks_confirmed"]}
            res["codec_bitrate"].append(row)
            print("codec", row, flush=True)
        res["decode"].append({"codec": codec, "kbps": 1000,
                              "decode_ms_per_frame": decode_ms_per_frame(WORK / f"clip_{codec}_1000k.mp4")})
        print("decode", res["decode"][-1], flush=True)

    for stride in STRIDES:
        sm = count(det, s, CLIP, gt_line, stride=stride)
        row = {"stride": stride, "processed_fps": round(30 / stride, 1), **score(sm, CLIP),
               "tracks": sm["tracks_confirmed"]}
        res["fps"].append(row)
        print("fps", row, flush=True)

    if FACTORY.exists():
        ref = count(det, s, FACTORY, FACTORY_LINE, stride=2)["count"]
        res["factory"].append({"variant": "original, 30 fps processed", "count": ref, "agreement": "reference"})
        for stride, fps in ((4, 15), (6, 10), (12, 5)):
            c = count(det, s, FACTORY, FACTORY_LINE, stride=stride)["count"]
            res["factory"].append({"variant": f"original, {fps} fps processed", "count": c,
                                   "agreement": round(c / ref, 3)})
            print("factory", res["factory"][-1], flush=True)
        for codec in ("h264", "h265"):
            for kbps in FACTORY_BITRATES_K:
                v = WORK / f"factory_{codec}_{kbps}k.mp4"
                real = encode(FACTORY, codec, kbps, v)
                c = count(det, s, v, FACTORY_LINE, stride=2)["count"]
                res["factory"].append({"variant": f"{codec} {real} kbit/s, 30 fps processed", "count": c,
                                       "agreement": round(c / ref, 3)})
                print("factory", res["factory"][-1], flush=True)

    (OUT / "video_experiments.json").write_text(json.dumps(res, indent=2))
    (OUT / "video_experiments.md").write_text(to_markdown(res))
    print(to_markdown(res))


def to_markdown(r: dict[str, list[dict[str, object]]]) -> str:
    md = ["## Codec x bitrate (8 s clip, 720x1280 @ 30 fps, ground truth 9 crossings)", "",
          "| codec | bitrate kbit/s | count | precision | recall | tracks |", "|---|---|---|---|---|---|"]
    md += [f"| {x['codec']} | {x['kbps']} | {x['count']} | {x['precision']} | {x['recall']} | {x['tracks']} |"
           for x in r["codec_bitrate"]]
    md += ["", "## Processed FPS (same clip)", "", "| processed fps | count | precision | recall | tracks |",
           "|---|---|---|---|---|"]
    md += [f"| {x['processed_fps']} | {x['count']} | {x['precision']} | {x['recall']} | {x['tracks']} |" for x in r["fps"]]
    md += ["", "## Decode cost (single core, ~1 Mbit/s, 720x1280)", "", "| codec | ms per frame |", "|---|---|"]
    md += [f"| {x['codec']} | {x['decode_ms_per_frame']} |" for x in r["decode"]]
    if r["factory"]:
        md += ["", "## Factory video (464x832 @ 60 fps, 3.9 min): agreement with the full-quality run, not accuracy",
               "", "| variant | count | vs reference |", "|---|---|---|"]
        md += [f"| {x['variant']} | {x['count']} | {x['agreement']} |" for x in r["factory"]]
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    main()

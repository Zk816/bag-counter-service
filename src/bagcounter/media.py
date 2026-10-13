from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from bagcounter.errors import InvalidVideoError, StorageError

PROBE_TIMEOUT_S = 30.0
FRAME_TIMEOUT_S = 30.0
MIN_FPS, MAX_FPS = 1.0, 240.0


def tools_available() -> dict[str, bool]:
    return {t: shutil.which(t) is not None for t in ("ffmpeg", "ffprobe")}


def probe(path: Path, *, max_seconds: float, max_pixels: int) -> dict[str, object]:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames:format=duration,format_name",
           "-of", "json", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=PROBE_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired as e:
        raise InvalidVideoError("video inspection timed out") from e
    except OSError as e:
        raise StorageError("ffprobe is not available") from e
    if out.returncode != 0:
        raise InvalidVideoError("file is not a readable video")
    try:
        data = json.loads(out.stdout)
        st = data["streams"][0]
        w, h = int(st["width"]), int(st["height"])
        rate = st.get("avg_frame_rate") or st.get("r_frame_rate") or "0/1"
        fps = float(Fraction(rate)) if rate != "0/0" else 0.0
        duration = float(data.get("format", {}).get("duration") or 0.0)
    except (ValueError, KeyError, IndexError, ZeroDivisionError) as e:
        raise InvalidVideoError("file has no usable video stream") from e
    if not MIN_FPS <= fps <= MAX_FPS:
        raise InvalidVideoError("video frame rate out of range", context={"fps": round(fps, 2)})
    if w < 32 or h < 32 or w * h > max_pixels:
        raise InvalidVideoError("video resolution out of range", context={"width": w, "height": h})
    if duration <= 0 or duration > max_seconds:
        raise InvalidVideoError("video duration out of range", context={"seconds": round(duration, 1)})
    return {"width": w, "height": h, "fps": round(fps, 3), "duration_s": round(duration, 2),
            "codec": st.get("codec_name")}


def extract_frame(path: Path, out_jpg: Path, at_s: float = 0.0) -> None:
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{at_s:.2f}", "-i", str(path), "-frames:v", "1", "-q:v", "3",
           str(out_jpg)]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=FRAME_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired as e:
        raise InvalidVideoError("frame extraction timed out") from e
    if out.returncode != 0 or not out_jpg.is_file():
        raise InvalidVideoError("could not decode a frame from the video")

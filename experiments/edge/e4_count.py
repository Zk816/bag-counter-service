"""Does the edge pipeline count the same bags as the original GPU model?

References: the original infer_video.py (YOLOv8x, PyTorch, GPU) with and without test-time
augmentation. Candidates: every OpenVINO model through edge/pipeline.py on 8 CPU cores. Same
tracker, same line, same rules — only the detector and runtime change.

Finding that shaped this script: the count is sensitive. On the 8 s clip the same YOLOv8x gives 7
with TTA and 4 without, and ByteTrack creates ~50 track IDs for a handful of bags. So agreement with
one reference is not accuracy. If edge/ground_truth.json exists ({"test_video.mp4": 41, ...},
counted by a person), errors are computed against it; otherwise only the spread is reported.
"""

import json
import os
import subprocess
import sys

from common import RESULTS, ROOT, VIDEOS, WEIGHTS, ov_dir, pin_cpu
from pipeline import count_video


def reference_count(video, tta: bool = True) -> int:
    """Original script on GPU, parsed from its summary line."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("REF_GPU", "5")}
    script = (ROOT / "experiments" / "legacy" / "infer_video.py").read_text().replace('os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")', "pass")
    if not tta:
        script = script.replace("augment=True", "augment=False")
    out = subprocess.run([sys.executable, "-c", script,
                          "--source", str(video)], capture_output=True, text=True, env=env, timeout=3600, check=True)
    line = next(l for l in out.stdout.splitlines() if "Unique bags counted" in l)
    return int(line.split(":")[1])


def main() -> None:
    gt_file = ROOT / "experiments" / "edge" / "ground_truth.json"
    results = {"reference": {}, "reference_no_tta": {}, "ground_truth": json.loads(gt_file.read_text()) if gt_file.exists() else None,
               "edge": []}
    for v in VIDEOS:
        results["reference"][v.name] = reference_count(v, tta=True)
        results["reference_no_tta"][v.name] = reference_count(v, tta=False)
        print(f"reference {v.name}: TTA {results['reference'][v.name]}, no TTA {results['reference_no_tta'][v.name]}", flush=True)
    pin_cpu()
    for name in WEIGHTS:
        for fmt in ("fp32", "int8"):
            d = ov_dir(name, fmt)
            if not d.exists():
                continue
            for v in VIDEOS:
                r = {"model": name, "format": fmt, **count_video(d, v)}
                r["reference"] = results["reference"][v.name]
                results["edge"].append(r)
                print(r, flush=True)
                (RESULTS / "counting.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

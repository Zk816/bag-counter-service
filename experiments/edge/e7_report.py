"""results/*.json -> edge/RESULTS.md."""

import json
from collections import defaultdict

from common import EDGE, RESULTS, THREADS


def load(name: str):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    lines = ["# Edge deployment results (CPU, OpenVINO)", "",
             f"All CPU numbers pinned to {THREADS} cores of the server (an edge-box budget). Input 640x640.", ""]
    bench = load("benchmark.json")
    if bench:
        lines += ["## Accuracy and speed per model and runtime", "",
                  "| model | runtime | mAP50 | mAP50-95 | latency p50 | p95 | throughput (OpenVINO streams) |",
                  "|---|---|---|---|---|---|---|"]
        for r in bench["rows"]:
            tp = f"{r['throughput_fps']:.1f} FPS" if "throughput_fps" in r else "-"
            lines.append(f"| {r['model']} | {r['runtime']} | {r['mAP50']:.3f} | {r['mAP50_95']:.3f} | "
                         f"{r['p50_ms']:.1f} ms | {r['p95_ms']:.1f} ms | {tp} |")
        lines += ["", "mAP on a fixed 1000-image subset of the validation split; INT8 calibrated on 300 training frames.", ""]
    cnt = load("counting.json")
    if cnt:
        gt = cnt.get("ground_truth")
        vids = list(cnt["reference"])
        lines += ["## Counting on the test videos", "",
                  "| model | format | " + " | ".join(vids) + " | FPS (8 cores) |", "|---" * (len(vids) + 3) + "|"]
        if gt:
            lines.append("| **human count** | | " + " | ".join(str(gt.get(v, "?")) for v in vids) + " | |")
        lines.append("| yolov8x PyTorch GPU + TTA (original) | | " + " | ".join(str(cnt["reference"][v]) for v in vids) + " | |")
        lines.append("| yolov8x PyTorch GPU, no TTA | | " + " | ".join(str(cnt["reference_no_tta"][v]) for v in vids) + " | |")
        by = defaultdict(dict)
        for r in cnt["edge"]:
            by[(r["model"], r["format"])][r["video"]] = r
        for (m, f), d in by.items():
            fps = sum(x["fps"] for x in d.values()) / len(d)
            lines.append(f"| {m} | OpenVINO {f} | " + " | ".join(str(d[v]["count"]) if v in d else "-" for v in vids)
                         + f" | {fps:.1f} |")
        lines += ["", "**Read with care.** These counts use the old fixed rule (horizontal line at 60% of the height, "
                  "both directions). test_video.mp4 is a handheld promo clip (moving camera, bags travel sideways) and "
                  "test_video2.mp4 an edited promo with scene cuts: neither is a valid counting test, so this table "
                  "only shows runtime agreement. On test_video3.mp4 the 60% line sits in the truck where workers "
                  "handle bags. The service's accuracy is measured instead against hand-labelled crossings with a "
                  "user-drawn line: see README (9/9 on the chute clip, tests/integration)."]
        lines.append("")
    rt = load("rtsp.json")
    if rt:
        lines += [f"## Live RTSP, {rt['duration_s']} s per run (MediaMTX, H.264 ~30 FPS per camera, newest-frame policy)", "",
                  "| model | cameras | processed FPS / camera | frames dropped | latency p50 | p95 | counts per camera |",
                  "|---|---|---|---|---|---|---|"]
        for r in rt["rows"]:
            fps = sum(r["fps_per_camera"]) / len(r["fps_per_camera"])
            lines.append(f"| {r['model']} INT8 | {r['cameras']} | {fps:.1f} | {100 * r['dropped_share']:.0f}% | "
                         f"{r['latency_ms_p50']:.0f} ms | {r['latency_ms_p95']:.0f} ms | {r['counts']} |")
        lines.append("")
    cpp = load("cpp_vs_python.json")
    if cpp:
        lines += ["## C++ vs Python (same INT8 model, same 900 frames of test_video2, 8 cores)", "",
                  "| model | C++ sync FPS | C++ async FPS | Python FPS | C++ pre / infer / post ms | "
                  "Python pre / infer / post ms |", "|---|---|---|---|---|---|"]
        for r in cpp:
            c, p = r["cpp_sync"]["ms"], r["python"]["ms"]
            lines.append(f"| {r['model']} | {r['cpp_sync']['fps']:.1f} | {r['cpp_async']['fps']:.1f} | "
                         f"{r['python']['fps']:.1f} | {c['pre']:.1f} / {c['infer_wait']:.1f} / {c['post']:.2f} | "
                         f"{p['pre']:.1f} / {p['infer']:.1f} / {p['post_track']:.1f} (incl. tracking) |")
        lines += ["", "C++ letterboxes into a u8 buffer and lets OpenVINO's PrePostProcessor do BGR->RGB, u8->f32 and "
                  "/255 inside the compiled model; async = two infer requests, the next frame is read and letterboxed "
                  "while the current one infers. An earlier version of this table showed C++ slower than Python: "
                  "e6_cpp.py passed 1280x720 for a 640x360 video, so C++ read four frames as one. Fixed (size from "
                  "ffprobe, frame count asserted).", ""]
    (EDGE / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

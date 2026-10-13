import json
import os
import subprocess

from common import CPU_CORES, RESULTS, ROOT, THREADS, WEIGHTS, ov_dir, pin_cpu
from pipeline import count_video

BIN = ROOT / "experiments" / "edge" / "cpp" / "build" / "detector"
VIDEO = ROOT / "test_video2.mp4"
FRAMES = 900


def video_size(path) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=60, check=True)
    w, h = out.stdout.strip().split(",")[:2]
    return int(w), int(h)


def run_cpp(model_dir, async_: bool) -> dict:
    import openvino
    libs = os.path.join(os.path.dirname(openvino.__file__), "libs")
    xml = next(model_dir.glob("*.xml"))
    cores = ",".join(map(str, CPU_CORES))
    w, h = video_size(VIDEO)
    cmd = (f"ffmpeg -loglevel error -i {VIDEO} -frames:v {FRAMES} -f rawvideo -pix_fmt bgr24 - | "
           f"taskset -c {cores} {BIN} {xml} {w} {h} {FRAMES} {THREADS} {int(async_)}")
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800, check=True,  # noqa: S602
                         env={**os.environ, "LD_LIBRARY_PATH": libs})
    res = json.loads(out.stdout.strip().splitlines()[-1])
    if res["frames"] != FRAMES:
        raise RuntimeError(f"C++ processed {res['frames']} frames, expected {FRAMES}: wrong frame size?")
    return res


def main() -> None:
    rows = []
    for name in WEIGHTS:
        d = ov_dir(name, "int8")
        if not d.exists():
            continue
        cpp_sync, cpp_async = run_cpp(d, False), run_cpp(d, True)
        pin_cpu()
        py = count_video(d, VIDEO)
        rows.append({"model": name, "cpp_sync": cpp_sync, "cpp_async": cpp_async,
                     "python": {"fps": py["fps"], "ms": py["ms"], "frames": py["frames"]}})
        print(rows[-1], flush=True)
        (RESULTS / "cpp_vs_python.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

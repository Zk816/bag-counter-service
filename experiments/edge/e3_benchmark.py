"""Accuracy and speed of every model x runtime, on 8 CPU cores.

Accuracy: mAP50 / mAP50-95 on the fixed 1000-image val subset (Ultralytics validator, CPU).
Speed: model-only latency for one 640x640 frame (p50/p95), and OpenVINO THROUGHPUT mode FPS
(several inference streams in parallel — the multi-camera case).
"""

import json
import time

import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

from common import IMGSZ, RESULTS, THREADS, VAL_YAML, WEIGHTS, ov_dir, pin_cpu
from pipeline import compile_model

RUNS = 200


def timed(fn, runs: int = RUNS) -> dict[str, float]:
    for _ in range(10):
        fn()
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        fn()
        t.append((time.perf_counter() - s) * 1000)
    return {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95))}


def ov_throughput(model_dir, seconds: float = 15.0) -> float:
    import openvino as ov
    compiled = compile_model(model_dir, hint="THROUGHPUT")
    n = compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
    queue = ov.AsyncInferQueue(compiled, n)
    x = np.random.rand(1, 3, IMGSZ, IMGSZ).astype(np.float32)
    done = 0
    def cb(*_):  # noqa: ANN002
        nonlocal done
        done += 1
    queue.set_callback(cb)
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        queue.start_async({0: x})
    queue.wait_all()
    return done / (time.perf_counter() - start)


def main() -> None:
    pin_cpu()
    torch.set_num_threads(THREADS)
    x = np.random.rand(1, 3, IMGSZ, IMGSZ).astype(np.float32)
    rows = []
    for name, w in WEIGHTS.items():
        if not w.exists():
            continue
        variants = {"pytorch_fp32": w, "onnxruntime_fp32": w.with_suffix(".onnx"),
                    "openvino_fp32": ov_dir(name, "fp32"), "openvino_int8": ov_dir(name, "int8")}
        for runtime, path in variants.items():
            if not path.exists():
                continue
            print(f"== {name} {runtime}", flush=True)
            m = YOLO(str(path), task="detect").val(data=str(VAL_YAML), imgsz=IMGSZ, batch=1 if "openvino" in runtime else 16,
                                                   device="cpu", plots=False, verbose=False)
            row = {"model": name, "runtime": runtime, "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map)}
            if runtime == "pytorch_fp32":
                net = YOLO(str(path)).model.eval().float()
                xt = torch.from_numpy(x)
                with torch.inference_mode():
                    row.update(timed(lambda: net(xt), runs=50 if name == "yolov8x" else RUNS))
            elif runtime == "onnxruntime_fp32":
                so = ort.SessionOptions(); so.intra_op_num_threads = THREADS
                sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
                inp = sess.get_inputs()[0].name
                row.update(timed(lambda: sess.run(None, {inp: x}), runs=50 if name == "yolov8x" else RUNS))
            else:
                req = compile_model(path).create_infer_request()
                row.update(timed(lambda: req.infer({0: x})))
                row["throughput_fps"] = ov_throughput(path)
            rows.append(row)
            print(row, flush=True)
            (RESULTS / "benchmark.json").write_text(json.dumps({"threads": THREADS, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()

# ADR 0001: Serve YOLOv8s INT8 on CPU with OpenVINO

**Status:** accepted · **Date:** 2026-10-12

## Context
The counter runs on an on-prem box at the factory. GPUs are expensive, hard to procure and to cool next to a
production line, and one camera at 15–30 FPS does not need one. Measured on 8 pinned CPU cores
(`experiments/edge/RESULTS.md`, 1000-image validation subset):

| model | runtime | mAP50 | detector latency p50 | file counting FPS |
|---|---|---|---|---|
| YOLOv8x | PyTorch FP32 | 0.963 | 223 ms | – |
| YOLOv8x | OpenVINO INT8 | 0.947 | 33 ms | 18 |
| **YOLOv8s** | **OpenVINO INT8** | **0.944** | **7.7 ms** | **62** |
| YOLOv8n | OpenVINO INT8 | 0.913 | 4.9 ms | 81 |

## Decision
Ship YOLOv8s, INT8 (NNCF post-training quantisation, 300 calibration frames), OpenVINO on CPU, with input
conversion (u8 BGR NHWC → f32 RGB NCHW / 255) compiled into the model with PrePostProcessor. Default
`PERFORMANCE_HINT=LATENCY`: on the stack test it cut detector p50 from ~45 ms to ~15 ms versus `THROUGHPUT` with
one camera plus one video job, because THROUGHPUT splits the cores into parallel streams.

## Consequences
- v8s gives up 1.9 mAP50 points against v8x and runs 3.4× faster: room for 4 cameras on 8 cores.
- v8n stays the fallback for weaker boxes (−3.1 points; packaged the same way with `scripts/package_model.py`).
- INT8 output depends on the CPU's instruction set, so the packaged self-test (ADR 0005) must pass on each new
  hardware type.

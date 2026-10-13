# ADR 0003: The runtime image contains no torch and no Ultralytics

**Status:** accepted · **Date:** 2026-10-12

## Context
Training uses Ultralytics (AGPL-3.0) on GPUs. The first inference scripts used it too, which pulled torch,
CUDA wheels and the AGPL tracker code into anything that served counts.

## Decision
- Inference: OpenVINO runtime only, with letterbox/decode/NMS written in `bagcounter/detect/openvino.py`.
- Tracking: ByteTrack implemented from the paper in `bagcounter/track/` (numpy + scipy).
- Training and export stay in `scripts/` and `experiments/`, with `requirements-train.txt`.

## Consequences
- Runtime image 1.18 GB (Python + OpenVINO + OpenCV + ffmpeg) instead of ~8 GB with torch + CUDA.
- We own the tracker: changes for counting (ids only for confirmed tracks, removal events for live
  finalisation) are in our code and tested.
- **Licensing is not solved by this alone.** The model weights were trained with Ultralytics, which states that
  models trained with it fall under AGPL-3.0 unless an Enterprise licence is bought. This repository is therefore
  published under AGPL-3.0. A closed commercial deployment needs either that licence or a detector retrained with
  a permissively licensed framework.

# ADR 0002: Live cameras keep only the newest frame

**Status:** accepted · **Date:** 2026-10-12

## Context
A camera delivers 25–30 FPS whether or not the box keeps up. A queue between decode and inference grows without
bound when inference is slower (CPU contention, a video job running, a burst of detections), and the count
then lags reality by minutes while memory grows.

## Decision
One reader thread per camera decodes continuously and keeps a single slot with the newest frame. The processing
loop samples at `target_fps` (15) and always takes the newest frame; older frames are dropped and counted
(`bagcounter_frames_dropped`).

## Consequences
- Latency is bounded (p95 ~110 ms measured on the demo camera), memory is constant.
- Dropped frames are the normal state, not an error (~50% at 30 FPS in / 15 FPS processed).
- Processed FPS has a floor below which bags are missed; it is measured in `experiments/results/video_experiments.md`
  and alerted on (`BagcounterProcessingTooSlow`).

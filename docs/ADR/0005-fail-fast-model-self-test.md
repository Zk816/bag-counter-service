# ADR 0005: The worker refuses to start unless the model reproduces a packaged answer

**Status:** accepted · **Date:** 2026-10-12

## Context
A counter that runs but counts wrong is worse than one that is down: nobody notices, and the numbers go into
shipping records. Typical silent causes: a corrupted model file, a different OpenVINO build, a CPU without the
INT8 instructions the model was tuned on.

## Decision
`scripts/package_model.py` stores, next to the model, a fixed frame and the boxes the model produced on it, plus
SHA-256 checksums of every file. At boot the worker verifies checksums, compiles the model, re-runs the frame and
compares boxes (IoU ≥ 0.9, score ± 0.05). Any failure: exit 1, no serving. The same self-test repeats every
5 minutes; a failure stops the worker and is alerted (`BagcounterSelftestFailing`).

## Consequences
- Moving to new hardware may require repackaging after an accuracy check; this is deliberate.
- `/readyz` is 503 until the self-test has passed, so a load balancer or operator sees the problem immediately.

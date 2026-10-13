#!/usr/bin/env bash
# Edge experiment end to end: train small models -> export -> benchmark -> counting -> RTSP -> C++ -> report.
# Each step keeps its results; re-running skips finished training and fresh exports.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="${EDGE_GPU:-5}"
PY=../../venv/bin/python
mkdir -p results
run() { echo "== $(date '+%F %T') $*"; "$PY" "$@" || echo "!! $* failed (continuing)"; }
{
# a training started by hand earlier: wait for it instead of starting a second one
while ps -eo args | grep -q "[e]1_train_small.py"; do sleep 60; done
run e1_train_small.py
run e2_export.py
cpp/build.sh
run e3_benchmark.py
run e4_count.py
run e5_rtsp.py
run e6_cpp.py
run e7_report.py
echo "done $(date)"
} 2>&1 | tee -a results/run_all.log

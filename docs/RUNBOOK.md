# Runbook

One section per alert in `deploy/prometheus/alerts.yml`: what the person on call sees, the likely causes, and the
first commands to run. All commands assume the repository root and the default compose project:

```bash
DC="docker compose -f deploy/docker-compose.yml"
```

Logs are JSON lines on stdout (`$DC logs worker --since 30m`). Every record carries `job_id` or `camera_id` when
it concerns one; identical errors are collapsed to one line per minute with a `repeated` count.

---

## BagcounterWorkerDown

**Symptom:** no worker heartbeat for 60 s. Video jobs stay queued, live counts stop, `/readyz` returns 503.

**Likely causes:** the worker crashed and is restart-looping (preflight failing); the job loop is stuck on a
pathological video (the heartbeat stops after `job_stall_s`); the host is out of memory.

**First commands:**
1. `$DC ps worker` – restarting? `unhealthy`?
2. `$DC logs worker --since 15m | grep -E '"level": "(CRITICAL|ERROR)"'` – a `preflight check` with `"ok": false`
   names the failing check.
3. `$DC exec worker python -m bagcounter.cli preflight; echo $?` – runs the same checks by hand.

A job that was running when the worker died is retried once on restart, then marked `failed / WORKER_CRASHED`.

## BagcounterSelftestFailing

**Symptom:** the model's output on the packaged self-test frame no longer matches the expectation recorded at
packaging time. The worker stops (exit 1) instead of producing counts it cannot vouch for.

**Likely causes:** corrupted model files on disk; a different OpenVINO version or a CPU without the instruction
set the INT8 kernels expect (moved to a different host); the model directory was swapped without repackaging.

**First commands:**
1. `$DC logs worker | grep -E 'self-test|checksum'`
2. `$DC exec worker sha256sum -c models/bag_v8s_int8/checksums.sha256` (run from `/app`)
3. `$DC exec worker python -c "import openvino; print(openvino.__version__)"` and `lscpu | grep -i -E 'avx512|vnni'`

Fix: redeploy the image (restores the files), or repackage the model on the new hardware with
`scripts/package_model.py` after verifying accuracy there.

## BagcounterCameraNoSignal

**Symptom:** a camera is `stalled` (no frames for `stall_after_s`) or `frozen` (frames arrive but the picture
does not change for `frozen_after_s`). Its bags are not being counted.

**Likely causes:** camera powered off / network down; wrong credentials after a camera reset; the NVR channel
froze; the lens is covered.

**First commands:**
1. Web UI → Live cameras: state and reconnect count.
2. `$DC exec worker ffprobe -v error -rtsp_transport tcp -i '<rtsp url>' -show_entries stream=codec_name,width,height`
3. `$DC logs worker --since 10m | grep <camera_id>`

The reader reconnects by itself with exponential backoff (1 s → 30 s); no restart is needed once the stream is back.

## BagcounterNoBagsCounted

**Symptom:** the camera is live and frames are processed, but no bag was counted for 30 min. This is the
silent-failure alarm: everything looks up, nothing is counted.

**Likely causes:** the line really stopped (check with the shift lead first); the camera was moved or zoomed
and the counting line no longer crosses the belt; the direction is set the wrong way; the product changed
and the model no longer detects it.

**First commands:**
1. Web UI → Live cameras → the live picture: are boxes drawn on the bags? Does the red line cross the belt?
2. Web UI → Set counting line → compare the arrow with the belt direction.
3. Grafana → "Median detection confidence": a drop points to the model, not the line.

## BagcounterProcessingTooSlow

**Symptom:** a live camera is processed at under 5 FPS. Bags that cross the line between two processed frames
can be missed or counted late.

**Likely causes:** too many cameras for the CPU budget; another process on the same cores; `infer_threads`
higher than the cores in `BAGCOUNTER_CPUSET`.

**First commands:**
1. Grafana → "Detector time per frame" (should be ~10–20 ms for v8s INT8 on 8 cores).
2. `docker stats --no-stream` – is the worker at its CPU limit?
3. Reduce `BAGCOUNTER_MAX_CAMERAS`, give the worker more cores, or switch to the v8n model (see README benchmarks).

## BagcounterHighLatency

**Symptom:** p95 time from frame decoded to count updated above 500 ms.

**Likely causes:** CPU saturation (see above); a video job running alongside live cameras on the same cores.

**First commands:** as for BagcounterProcessingTooSlow; check `bagcounter_queue_depth` for video jobs competing for CPU.

## BagcounterTrackFragmentation

**Symptom:** confirmed tracks per counted bag doubled versus the same hour yesterday.

**Likely causes:** the camera moved; lighting changed (a lamp failed, sunlight through a door); dust on the lens;
the belt runs faster. Fragmented tracks are the main way double counts and misses happen.

**First commands:**
1. Compare the live picture with yesterday's event snapshots (`GET /v1/cameras/<id>/events`).
2. Grafana → "Median detection confidence" for the same camera.

## BagcounterLowDetectionConfidence

**Symptom:** median detector score on a camera under 0.4 for 30 min.

**Likely causes:** a new bag design or product; very different lighting; out-of-focus camera. The model was
trained on one product family; a new one needs new training data (`scripts/harvest_crops.py` collects it).

**First commands:** look at the live picture; collect 10–20 minutes of video and run `bagcounter count` offline to
see where detections fail.

## BagcounterDiskLow

**Symptom:** under 5 GB free on the data volume. Under `min_free_disk_bytes` (2 GB) uploads are refused with 503.

**Likely causes:** many large annotated videos inside the retention window; retention set too long.

**First commands:**
1. `$DC exec worker du -sh /data/uploads /data/outputs /data/cameras`
2. Lower `BAGCOUNTER_RETAIN_OUTPUTS_S` / `BAGCOUNTER_RETAIN_EVENTS_S`; the reaper runs every minute.

## BagcounterJobsFailing

**Symptom:** more than 2 video jobs failed in an hour.

**First commands:**
1. `curl -s localhost:8080/v1/jobs?limit=20 | jq '.[] | select(.status=="failed") | .error'` – the `code` tells
   the class: `INPUT_INVALID_VIDEO` (bad file, client side), `WORKER_CRASHED` (crashed the worker twice: keep the file
   for investigation), `JOB_TIMEOUT`, `MODEL_NOT_READY`.
2. `$DC logs worker | grep <job_id>`

## BagcounterQueueBacklog

**Symptom:** more than 10 video jobs waiting for 15 min. New submissions get 429 at `max_queued_jobs`.

**First commands:** check whether a single very long video is running (`/v1/jobs`); live cameras share the same
CPU, so a backlog during a shift is expected. Process long archives off-shift.

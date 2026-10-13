from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

_LATENCY_BUCKETS = (0.002, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0)

STAGE_SECONDS = Histogram("bagcounter_stage_seconds", "Per-frame time of a pipeline stage", ["stage", "stream"],
                          buckets=_LATENCY_BUCKETS)
FRAME_LATENCY = Histogram("bagcounter_frame_latency_seconds", "Camera frame decoded -> count updated", ["stream"],
                          buckets=_LATENCY_BUCKETS)
FRAMES_PROCESSED = Counter("bagcounter_frames_processed_total", "Frames through the pipeline", ["stream"])
FRAMES_DECODED = Gauge("bagcounter_frames_decoded", "Frames decoded by the camera reader since start", ["stream"])
FRAMES_DROPPED = Gauge("bagcounter_frames_dropped", "Camera frames replaced by a newer one (newest-frame policy)",
                       ["stream"])
BAGS_COUNTED = Counter("bagcounter_bags_counted_total", "Count events", ["stream", "late"])
TRACKS_CONFIRMED = Counter("bagcounter_tracks_confirmed_total", "Confirmed tracks (fragmentation = tracks/bags)",
                           ["stream"])
DETECTION_SCORE = Histogram("bagcounter_detection_score", "Scores of tracked boxes (drift signal)", ["stream"],
                            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
STREAM_UP = Gauge("bagcounter_stream_state", "1 for the current state of each camera", ["stream", "state"])
STREAM_RECONNECTS = Gauge("bagcounter_stream_reconnects", "Reconnects since the camera started", ["stream"])
LAST_FRAME_TS = Gauge("bagcounter_stream_last_frame_timestamp_seconds", "Unix time of the last processed frame",
                      ["stream"])

JOBS = Counter("bagcounter_jobs_total", "Finished video jobs", ["outcome"])
JOB_SECONDS = Histogram("bagcounter_job_seconds", "Video job wall time", buckets=(5, 15, 30, 60, 120, 300, 600, 1800,
                                                                                3600, 10800))
QUEUE_DEPTH = Gauge("bagcounter_queue_depth", "Video jobs waiting")
WORKER_HEARTBEAT = Gauge("bagcounter_worker_heartbeat_timestamp_seconds", "Last worker loop iteration")
SELFTEST_OK = Gauge("bagcounter_selftest_ok", "1 if the last model self-test passed")
SELFTEST_TS = Gauge("bagcounter_selftest_last_success_timestamp_seconds", "Unix time of the last passing self-test")
DISK_FREE = Gauge("bagcounter_disk_free_bytes", "Free bytes on the data volume")
DATA_BYTES = Gauge("bagcounter_data_bytes", "Bytes used under data/", ["kind"])
REAPED = Counter("bagcounter_files_reaped_total", "Files deleted by retention", ["kind"])
HTTP_REQUESTS = Counter("bagcounter_http_requests_total", "HTTP requests", ["route", "status"])
HTTP_SECONDS = Histogram("bagcounter_http_request_seconds", "HTTP request time", ["route"])
MODEL = Info("bagcounter_model", "Model serving in this process")
CAMERA_INFO = Gauge("bagcounter_camera_info", "1 per running camera; maps camera id to name", ["stream", "name"])

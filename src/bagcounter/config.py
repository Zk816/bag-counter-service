from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BAGCOUNTER_", env_file=".env", extra="forbid")

    data_dir: Path = ROOT / "data"
    max_upload_bytes: int = Field(2 * 1024**3, gt=0)
    retain_uploads_s: int = Field(24 * 3600, gt=0)
    retain_outputs_s: int = Field(7 * 24 * 3600, gt=0)
    retain_events_s: int = Field(90 * 24 * 3600, gt=0)
    min_free_disk_bytes: int = Field(2 * 1024**3, ge=0)

    model_dir: Path = ROOT / "models" / "bag_v8s_int8"
    infer_threads: int = Field(8, ge=1, le=256)
    ov_performance_hint: str = Field("LATENCY", pattern="^(LATENCY|THROUGHPUT)$")
    conf_threshold: float = Field(0.1, gt=0, lt=1)
    nms_iou: float = Field(0.5, gt=0, lt=1)
    max_detections: int = Field(300, ge=1)
    selftest_interval_s: float = Field(300.0, gt=0)

    track_high_thresh: float = Field(0.25, gt=0, lt=1)
    track_low_thresh: float = Field(0.1, gt=0, lt=1)
    new_track_thresh: float = Field(0.25, gt=0, lt=1)
    track_buffer_s: float = Field(1.0, gt=0)
    match_thresh: float = Field(0.8, gt=0, le=1)

    min_hits: int = Field(3, ge=1)
    line_margin: float = Field(0.05, ge=0, le=0.5)

    max_queued_jobs: int = Field(20, ge=1)
    job_timeout_s: float = Field(3 * 3600.0, gt=0)
    worker_poll_s: float = Field(0.5, gt=0)
    max_video_seconds: float = Field(4 * 3600.0, gt=0)
    max_video_pixels: int = Field(3840 * 2160, gt=0)
    snapshot_max_side: int = Field(320, ge=32)
    job_stall_s: float = Field(120.0, gt=0)
    worker_metrics_port: int = Field(9101, ge=1, le=65535)
    reap_interval_s: float = Field(60.0, gt=0)
    heartbeat_stale_s: float = Field(30.0, gt=0)

    max_cameras: int = Field(4, ge=0)
    rtsp_open_timeout_s: float = Field(10.0, gt=0)
    rtsp_read_timeout_s: float = Field(5.0, gt=0)
    reconnect_backoff_min_s: float = Field(1.0, gt=0)
    reconnect_backoff_max_s: float = Field(30.0, gt=0)
    stall_after_s: float = Field(10.0, gt=0)
    frozen_after_s: float = Field(30.0, gt=0)
    target_fps: float = Field(15.0, gt=0)
    preview_fps: float = Field(5.0, gt=0)
    camera_state_flush_s: float = Field(5.0, gt=0)

    host: str = "0.0.0.0"  # noqa: S104
    port: int = Field(8080, ge=1, le=65535)
    log_level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")
    error_log_window_s: float = Field(60.0, gt=0)

    @model_validator(mode="after")
    def _consistent(self) -> Settings:
        if self.track_low_thresh > self.track_high_thresh:
            raise ValueError("track_low_thresh must be <= track_high_thresh")
        if self.conf_threshold > self.track_low_thresh:
            raise ValueError("conf_threshold above track_low_thresh would hide low-score boxes from the tracker")
        if self.reconnect_backoff_min_s > self.reconnect_backoff_max_s:
            raise ValueError("reconnect_backoff_min_s must be <= reconnect_backoff_max_s")
        if self.stall_after_s <= self.rtsp_read_timeout_s:
            raise ValueError("stall_after_s must exceed rtsp_read_timeout_s")
        return self

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "bagcounter.sqlite3"

    def redacted(self) -> dict[str, object]:
        return {k: str(v) if isinstance(v, Path) else v for k, v in self.model_dump().items()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

MAX_RECORD_CHARS = 8192

request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
job_id: ContextVar[str | None] = ContextVar("job_id", default=None)
camera_id: ContextVar[str | None] = ContextVar("camera_id", default=None)

_STD = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, var in (("request_id", request_id), ("job_id", job_id), ("camera_id", camera_id)):
            v = var.get()
            if v:
                out[k] = v
        for k, v in vars(record).items():
            if k not in _STD and not k.startswith("_"):
                out[k] = v
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        s = json.dumps(out, default=str)
        return s if len(s) <= MAX_RECORD_CHARS else s[: MAX_RECORD_CHARS - 20] + '..."truncated":true}'


class DedupFilter(logging.Filter):
    MAX_KEYS = 1024

    def __init__(self, window_s: float) -> None:
        super().__init__()
        self.window_s = window_s
        self._seen: dict[tuple[str, int, str], tuple[float, int]] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING:
            return True
        exc = record.exc_info[0].__name__ if record.exc_info and record.exc_info[0] else ""
        key = (record.name, record.lineno, exc or (record.msg if isinstance(record.msg, str) else ""))
        now = time.monotonic()
        first, n = self._seen.get(key, (0.0, 0))
        if now - first < self.window_s:
            self._seen[key] = (first, n + 1)
            return False
        if n:
            record.repeated = n
        if len(self._seen) >= self.MAX_KEYS:
            self._seen.clear()
        self._seen[key] = (now, 0)
        return True


def setup_logging(level: str, dedup_window_s: float) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(DedupFilter(dedup_window_s))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("uvicorn.access",):
        logging.getLogger(noisy).setLevel(logging.WARNING)

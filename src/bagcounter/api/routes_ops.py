from __future__ import annotations

import shutil
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from bagcounter.api.deps import AppState, get_state
from bagcounter.errors import StorageError
from bagcounter.media import tools_available

router = APIRouter(tags=["ops"])


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness: the process answers. Cheap on purpose; dependencies are /readyz's job."""
    return {"status": "ok"}


@router.get("/readyz")
def readyz(st: Annotated[AppState, Depends(get_state)]) -> JSONResponse:
    """Readiness: 503 unless every dependency is healthy, with the status of each one."""
    s = st.settings
    checks: dict[str, dict[str, Any]] = {}
    try:
        st.store.ping()
        checks["database"] = {"ok": True}
    except StorageError as e:
        checks["database"] = {"ok": False, "error": e.message}
    try:
        hb = st.store.get_meta("worker_heartbeat")
        age = time.time() - hb[1] if hb else None
        checks["worker"] = {"ok": age is not None and age < s.heartbeat_stale_s,
                            "heartbeat_age_s": round(age, 1) if age is not None else None}
        stt = st.store.get_meta("selftest")
        checks["model_selftest"] = {"ok": bool(stt and stt[0].get("ok")),
                                    "detail": stt[0] if stt else "never ran"}
    except StorageError as e:
        checks["worker"] = checks["model_selftest"] = {"ok": False, "error": e.message}
    free = shutil.disk_usage(s.data_dir).free
    checks["disk"] = {"ok": free >= s.min_free_disk_bytes, "free_bytes": free}
    tools = tools_available()
    checks["ffmpeg"] = {"ok": all(tools.values()), **tools}
    ok = all(c["ok"] for c in checks.values())
    return JSONResponse({"ready": ok, "checks": checks}, status_code=200 if ok else 503)


@router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/version")
def version(st: Annotated[AppState, Depends(get_state)]) -> dict[str, Any]:
    m = st.model
    return {"service": "bagcounter", "version": "1.0.0", "model": {"name": m.name, "version": m.version,
                                                                  "metrics": m.meta.get("metrics")},
            "started_at": st.started_at, "limits": {"max_upload_bytes": st.settings.max_upload_bytes,
                                                    "max_video_seconds": st.settings.max_video_seconds,
                                                    "max_cameras": st.settings.max_cameras}}

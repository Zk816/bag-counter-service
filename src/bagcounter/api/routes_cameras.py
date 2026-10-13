from __future__ import annotations

import asyncio
import shutil
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any
from urllib.parse import urlsplit, urlunsplit

import anyio
from fastapi import APIRouter, Depends, Path, Query, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from bagcounter.api.deps import AppState, get_state
from bagcounter.api.routes_jobs import ID, LineIn
from bagcounter.cameras import camera_dir
from bagcounter.errors import NotFoundError
from bagcounter.geometry import line_from_dict
from bagcounter.store import Camera

router = APIRouter(prefix="/v1/cameras", tags=["cameras"])

MJPEG_MAX_S = 15 * 60
MAX_RANGE_S = 92 * 24 * 3600
BOUNDARY = "frame"


def mask_url(url: str) -> str:
    """Never return or log camera passwords."""
    parts = urlsplit(url)
    if parts.password:
        host = parts.hostname or ""
        netloc = f"{parts.username}:***@{host}" + (f":{parts.port}" if parts.port else "")
        return urlunsplit(parts._replace(netloc=netloc))
    return url


def _check_url(v: str) -> str:
    parts = urlsplit(v)
    if parts.scheme not in ("rtsp", "rtsps", "http", "https") or not parts.hostname:
        raise ValueError("url must be rtsp://, rtsps://, http:// or https:// with a host")
    return v


class CameraIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    url: str = Field(min_length=8, max_length=500)
    line: LineIn | None = None
    enabled: bool = True

    _url = field_validator("url")(_check_url)


class CameraPatch(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=80)
    url: str | None = Field(None, min_length=8, max_length=500)
    line: LineIn | None = None
    enabled: bool | None = None

    @field_validator("url")
    @classmethod
    def _url(cls, v: str | None) -> str | None:
        return None if v is None else _check_url(v)


def camera_view(c: Camera, stale_after_s: float) -> dict[str, Any]:
    stale = c.enabled and (c.updated_at is None or time.time() - c.updated_at > stale_after_s)
    return {"camera_id": c.id, "name": c.name, "url": mask_url(c.url), "line": c.line, "enabled": c.enabled,
            "state": "unknown" if stale else c.state,
            "count_total": c.count_total, "status": c.status, "updated_at": c.updated_at,
            "preview_url": f"/v1/cameras/{c.id}/preview.jpg", "live_url": f"/v1/cameras/{c.id}/live.mjpeg"}


@router.get("")
def list_cameras(st: Annotated[AppState, Depends(get_state)]) -> list[dict[str, Any]]:
    return [camera_view(c, st.settings.heartbeat_stale_s) for c in st.store.list_cameras()]


@router.post("", status_code=201)
def add_camera(body: CameraIn, st: Annotated[AppState, Depends(get_state)]) -> dict[str, Any]:
    """Raises: CameraLimitError, InvalidLineError."""
    line = line_from_dict(body.line.model_dump()).to_dict() if body.line else None
    c = st.store.add_camera(body.name, body.url, line, st.settings.max_cameras)
    if not body.enabled:
        c = st.store.update_camera(c.id, enabled=False)
    return camera_view(c, st.settings.heartbeat_stale_s)


@router.get("/{camera_id}")
def get_camera(camera_id: ID, st: Annotated[AppState, Depends(get_state)]) -> dict[str, Any]:
    return camera_view(st.store.get_camera(camera_id), st.settings.heartbeat_stale_s)


@router.patch("/{camera_id}")
def patch_camera(camera_id: ID, body: CameraPatch, st: Annotated[AppState, Depends(get_state)]) -> dict[str, Any]:
    line = line_from_dict(body.line.model_dump()).to_dict() if body.line else None
    return camera_view(st.store.update_camera(camera_id, name=body.name, url=body.url, line=line,
                                              enabled=body.enabled), st.settings.heartbeat_stale_s)


@router.delete("/{camera_id}", status_code=204)
def delete_camera(camera_id: ID, st: Annotated[AppState, Depends(get_state)]) -> Response:
    st.store.delete_camera(camera_id)
    st.store.delete_camera_events(camera_id)
    shutil.rmtree(camera_dir(st.settings, camera_id), ignore_errors=True)
    return Response(status_code=204)


@router.get("/{camera_id}/preview.jpg")
def preview(camera_id: ID, st: Annotated[AppState, Depends(get_state)]) -> FileResponse:
    st.store.get_camera(camera_id)
    p = camera_dir(st.settings, camera_id) / "preview.jpg"
    if not p.is_file():
        raise NotFoundError("no frame received from this camera yet")
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/{camera_id}/live.mjpeg")
async def live(camera_id: ID, request: Request, st: Annotated[AppState, Depends(get_state)]) -> StreamingResponse:
    """Annotated preview as multipart MJPEG (works in an <img> tag). Bounded: closes after MJPEG_MAX_S."""
    await anyio.to_thread.run_sync(st.store.get_camera, camera_id)
    p = camera_dir(st.settings, camera_id) / "preview.jpg"
    interval = 1.0 / st.settings.preview_fps

    async def frames() -> AsyncIterator[bytes]:
        last_mtime = 0.0
        end = time.monotonic() + MJPEG_MAX_S
        while time.monotonic() < end and not await request.is_disconnected():
            try:
                mtime = p.stat().st_mtime
            except FileNotFoundError:
                mtime = last_mtime
            if mtime != last_mtime:
                last_mtime = mtime
                data = await anyio.Path(p).read_bytes()
                yield (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n\r\n"
                       .encode() + data + b"\r\n")
            await asyncio.sleep(interval)

    return StreamingResponse(frames(), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                             headers={"Cache-Control": "no-store"})


def _range(since: float | None, until: float | None, default_s: float) -> tuple[float, float]:
    until = until or time.time()
    since = since if since is not None else until - default_s
    return max(since, until - MAX_RANGE_S), until


@router.get("/{camera_id}/counts")
def counts(camera_id: ID, st: Annotated[AppState, Depends(get_state)],
           since: Annotated[float | None, Query(description="unix seconds")] = None,
           until: Annotated[float | None, Query(description="unix seconds")] = None,
           bucket_s: Annotated[int, Query(ge=60, le=86400)] = 3600) -> dict[str, Any]:
    """Bag counts per time bucket (default: last 24 h by hour)."""
    st.store.get_camera(camera_id)
    a, b = _range(since, until, 86400)
    buckets = st.store.count_buckets(camera_id, a, b, bucket_s)
    return {"camera_id": camera_id, "since": a, "until": b, "bucket_s": bucket_s,
            "total": sum(x["count"] for x in buckets), "buckets": buckets}


@router.get("/{camera_id}/events")
def events(camera_id: ID, st: Annotated[AppState, Depends(get_state)],
           since: float | None = None, until: float | None = None,
           limit: Annotated[int, Query(ge=1, le=1000)] = 100) -> list[dict[str, Any]]:
    st.store.get_camera(camera_id)
    a, b = _range(since, until, 86400)
    rows = st.store.list_events(camera_id, a, b, limit)
    for r in rows:
        r["snapshot_url"] = f"/v1/cameras/{camera_id}/{r['snapshot'].split('/', 2)[2]}" if r["snapshot"] else None
    return rows


@router.get("/{camera_id}/snapshots/{day}/{name}")
def snapshot(camera_id: ID, day: Annotated[str, Path(pattern=r"^\d{8}$")],
             name: Annotated[str, Path(pattern=r"^\d+_\d+\.jpg$")],
             st: Annotated[AppState, Depends(get_state)]) -> FileResponse:
    p = camera_dir(st.settings, camera_id) / "snapshots" / day / name
    if not p.is_file():
        raise NotFoundError("snapshot not found (it may have been deleted by retention)")
    return FileResponse(p, media_type="image/jpeg")

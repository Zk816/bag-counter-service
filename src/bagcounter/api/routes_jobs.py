from __future__ import annotations

import re
import shutil
from typing import Annotated, Any

import anyio
from fastapi import APIRouter, Depends, Path, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from bagcounter.api.deps import AppState, get_state
from bagcounter.errors import ConflictError, DiskPressureError, InvalidInputError, NotFoundError, PayloadTooLargeError
from bagcounter.geometry import line_from_dict
from bagcounter.media import extract_frame, probe
from bagcounter.store import Job, new_id

router = APIRouter(prefix="/v1", tags=["jobs"])

ID = Annotated[str, Path(pattern=r"^[0-9a-f]{16}$")]
OUTPUT_FILE = re.compile(r"^(annotated\.mp4|events\.csv|report\.json|snapshots/event_\d{5}\.jpg)$")
MEDIA = {".mp4": "video/mp4", ".csv": "text/csv", ".json": "application/json", ".jpg": "image/jpeg"}


class LineIn(BaseModel):
    ax: float = Field(ge=0, le=1)
    ay: float = Field(ge=0, le=1)
    bx: float = Field(ge=0, le=1)
    by: float = Field(ge=0, le=1)
    direction: str = Field("positive", pattern="^(positive|negative|both)$")


class JobIn(BaseModel):
    upload_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    line: LineIn
    annotate: bool = True


def job_view(j: Job) -> dict[str, Any]:
    files = None
    if j.status == "succeeded" and not j.outputs_expired:
        base = f"/v1/jobs/{j.id}/files"
        files = {"report_json": f"{base}/report.json", "events_csv": f"{base}/events.csv",
                 "annotated_video": f"{base}/annotated.mp4" if j.annotate else None}
    return {"job_id": j.id, "status": j.status, "progress": round(j.progress, 3), "count": j.count,
            "filename": j.filename, "line": j.line, "created_at": j.created_at, "started_at": j.started_at,
            "finished_at": j.finished_at, "attempts": j.attempts, "model_version": j.model_version,
            "error": {"code": j.error_code, "message": j.error_message} if j.error_code else None,
            "summary": j.summary, "files": files, "outputs_expired": j.outputs_expired,
            "poll_url": f"/v1/jobs/{j.id}"}


@router.post("/uploads", status_code=201)
async def upload(request: Request, st: Annotated[AppState, Depends(get_state)],
                 filename: Annotated[str, Query(max_length=200)] = "video.mp4") -> dict[str, Any]:
    """Raw video bytes in the body (no multipart): streamed to disk, size cap enforced while streaming."""
    s = st.settings
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > s.max_upload_bytes:
        raise PayloadTooLargeError("video exceeds the upload limit", context={"max_bytes": s.max_upload_bytes})
    free = await anyio.to_thread.run_sync(lambda: shutil.disk_usage(s.data_dir).free)
    if free < s.min_free_disk_bytes:
        raise DiskPressureError("not enough free disk space to accept a video", context={"free_bytes": free})

    uid = new_id()
    part = s.uploads_dir / f"{uid}.part"
    final = s.uploads_dir / f"{uid}.video"
    preview = s.uploads_dir / f"{uid}.jpg"
    size = 0
    try:
        f = await anyio.open_file(part, "wb")
        try:
            async for chunk in request.stream():
                size += len(chunk)
                if size > s.max_upload_bytes:
                    raise PayloadTooLargeError("video exceeds the upload limit",
                                               context={"max_bytes": s.max_upload_bytes})
                await f.write(chunk)
        finally:
            await f.aclose()
        if size == 0:
            raise InvalidInputError("empty body: send the video bytes as the request body")
        await anyio.to_thread.run_sync(part.rename, final)
        info = await anyio.to_thread.run_sync(
            lambda: probe(final, max_seconds=s.max_video_seconds, max_pixels=s.max_video_pixels))
        await anyio.to_thread.run_sync(lambda: extract_frame(final, preview))
        safe_name = re.sub(r"[^\w.\- ]", "_", filename)[:200] or "video.mp4"
        await anyio.to_thread.run_sync(lambda: st.store.add_upload(uid, safe_name, size, info))
    except BaseException:
        for p in (part, final, preview):
            p.unlink(missing_ok=True)
        raise
    return {"upload_id": uid, "bytes": size, "video": info, "preview_url": f"/v1/uploads/{uid}/preview.jpg"}


@router.get("/uploads/{upload_id}/preview.jpg")
def upload_preview(upload_id: ID, st: Annotated[AppState, Depends(get_state)]) -> FileResponse:
    st.store.get_upload(upload_id)
    p = st.settings.uploads_dir / f"{upload_id}.jpg"
    if not p.is_file():
        raise NotFoundError("preview not available")
    return FileResponse(p, media_type="image/jpeg")


@router.post("/jobs", status_code=202)
def submit(body: JobIn, st: Annotated[AppState, Depends(get_state)]) -> JSONResponse:
    """Queue a counting job for an upload."""
    line = line_from_dict(body.line.model_dump())
    job = st.store.create_job(body.upload_id, line.to_dict(), body.annotate, st.settings.max_queued_jobs)
    return JSONResponse(job_view(job), status_code=202, headers={"Location": f"/v1/jobs/{job.id}"})


@router.get("/jobs")
def list_jobs(st: Annotated[AppState, Depends(get_state)],
              limit: Annotated[int, Query(ge=1, le=200)] = 20) -> list[dict[str, Any]]:
    return [job_view(j) for j in st.store.list_jobs(limit)]


@router.get("/jobs/{job_id}")
def get_job(job_id: ID, st: Annotated[AppState, Depends(get_state)]) -> dict[str, Any]:
    return job_view(st.store.get_job(job_id))


@router.post("/jobs/{job_id}/cancel", status_code=202)
def cancel(job_id: ID, st: Annotated[AppState, Depends(get_state)]) -> dict[str, Any]:
    return job_view(st.store.request_cancel(job_id))


@router.get("/jobs/{job_id}/files/{name:path}")
def job_file(job_id: ID, name: str, st: Annotated[AppState, Depends(get_state)]) -> FileResponse:
    """Download one output of a finished job."""
    if not OUTPUT_FILE.match(name):
        raise NotFoundError("no such output", context={"name": name[:100]})
    job = st.store.get_job(job_id)
    if job.status != "succeeded":
        raise ConflictError("job has not finished successfully", context={"status": job.status})
    if job.outputs_expired:
        raise NotFoundError("outputs deleted by the retention policy")
    p = st.settings.outputs_dir / job_id / name
    if not p.is_file():
        raise NotFoundError("output not found", context={"name": name})
    download = name if name in ("events.csv", "report.json") else None
    return FileResponse(p, media_type=MEDIA[p.suffix], filename=f"{job_id}_{download}" if download else None)

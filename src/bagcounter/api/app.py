from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from bagcounter.api import routes_cameras, routes_jobs, routes_ops
from bagcounter.api.deps import AppState
from bagcounter.config import Settings, get_settings
from bagcounter.detect.openvino import verify_model_dir
from bagcounter.errors import AppError
from bagcounter.logging import request_id, setup_logging
from bagcounter.metrics import HTTP_REQUESTS, HTTP_SECONDS
from bagcounter.store import Store

log = logging.getLogger("bagcounter.api")
STATIC = Path(__file__).parent / "static"


def problem(status: int, code: str, title: str, *, retryable: bool = False, rid: str | None = None,
            context: dict[str, object] | None = None, headers: dict[str, str] | None = None) -> JSONResponse:
    body: dict[str, object] = {"type": f"https://bagcounter/errors/{code.lower()}", "title": title, "status": status,
                               "code": code, "retryable": retryable, "request_id": rid}
    if context:
        body["context"] = context
    return JSONResponse(body, status_code=status, media_type="application/problem+json", headers=headers)


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    setup_logging(s.log_level, s.error_log_window_s)
    for d in (s.data_dir, s.uploads_dir, s.outputs_dir):
        d.mkdir(parents=True, exist_ok=True)
    state = AppState(settings=s, store=Store(s.db_path), model=verify_model_dir(s.model_dir), started_at=time.time())
    app = FastAPI(title="Bag counter", version="1.0.0", docs_url="/docs", redoc_url=None)
    app.state.app = state
    log.info("api starting", extra={"config": s.redacted(), "model": f"{state.model.name}:{state.model.version}"})

    @app.middleware("http")
    async def context(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id.set(rid[:64])
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id.reset(token)
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        HTTP_REQUESTS.labels(route=path, status=str(response.status_code)).inc()
        HTTP_SECONDS.labels(route=path).observe(time.perf_counter() - started)
        response.headers["X-Request-Id"] = rid[:64]
        response.headers["X-Model-Version"] = f"{state.model.name}:{state.model.version}"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.exception_handler(AppError)
    async def app_error(request: Request, e: AppError) -> JSONResponse:
        if e.http_status >= 500:
            log.error("request failed", exc_info=e, extra={"code": e.code})
        headers = {"Retry-After": "30"} if e.retryable else None
        return problem(e.http_status, e.code, e.message, retryable=e.retryable, rid=request_id.get(),
                       context=e.context, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, e: RequestValidationError) -> JSONResponse:
        errs = [{"loc": list(x.get("loc", ())), "msg": x.get("msg")} for x in e.errors()][:10]
        return problem(422, "INPUT_INVALID", "request validation failed", rid=request_id.get(),
                       context={"errors": errs})

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, e: StarletteHTTPException) -> JSONResponse:
        return problem(e.status_code, "NOT_FOUND" if e.status_code == 404 else "HTTP_ERROR", str(e.detail),
                       rid=request_id.get())

    @app.exception_handler(Exception)
    async def unhandled(request: Request, e: Exception) -> JSONResponse:
        log.error("unhandled error", exc_info=e)
        return problem(500, "INTERNAL", "internal error", rid=request_id.get())

    app.include_router(routes_ops.router)
    app.include_router(routes_jobs.router)
    app.include_router(routes_cameras.router)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app

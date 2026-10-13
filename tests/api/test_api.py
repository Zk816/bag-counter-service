from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bagcounter.api.app import create_app
from bagcounter.config import Settings

DATA = Path(__file__).parents[1] / "data"
CLIP = (DATA / "clip_8s.mp4").read_bytes()
LINE = {"ax": 0.3, "ay": 0.35, "bx": 0.7, "by": 0.35, "direction": "positive"}


@pytest.fixture
def client(tmp_path):
    s = Settings(data_dir=tmp_path / "data", max_upload_bytes=5_000_000, max_queued_jobs=2)
    with TestClient(create_app(s)) as c:
        yield c


def assert_problem(r, status, code):
    assert r.status_code == status, r.text
    assert r.headers["content-type"].startswith("application/problem+json")
    body = r.json()
    assert body["code"] == code and body["status"] == status
    assert {"type", "title", "retryable", "request_id"} <= set(body)
    assert r.headers["x-request-id"] == body["request_id"]
    return body


def upload(client, data=CLIP):
    return client.post("/v1/uploads?filename=clip.mp4", content=data)


def test_every_response_carries_request_and_model_headers(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.headers["x-request-id"] and r.headers["x-model-version"].startswith("bag-")


def test_upload_returns_video_info_and_preview(client):
    r = upload(client)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["video"]["width"] == 720 and body["video"]["height"] == 1280
    assert client.get(body["preview_url"]).headers["content-type"] == "image/jpeg"


def test_upload_rejects_non_video_without_keeping_it(client, tmp_path):
    assert_problem(upload(client, b"definitely not a video" * 100), 400, "INPUT_INVALID_VIDEO")
    assert list((tmp_path / "data" / "uploads").iterdir()) == []


def test_upload_rejects_empty_body(client):
    assert_problem(upload(client, b""), 400, "INPUT_INVALID")


def test_upload_over_limit_is_rejected_during_the_stream(client, tmp_path):
    def chunks():
        for _ in range(10):
            yield b"\0" * 1_000_000
    assert_problem(client.post("/v1/uploads", content=chunks()), 413, "PAYLOAD_TOO_LARGE")
    assert list((tmp_path / "data" / "uploads").iterdir()) == []


def test_job_lifecycle_until_queued_and_cancel(client):
    uid = upload(client).json()["upload_id"]
    r = client.post("/v1/jobs", json={"upload_id": uid, "line": LINE})
    assert r.status_code == 202 and r.headers["location"] == f"/v1/jobs/{r.json()['job_id']}"
    job = r.json()
    assert job["status"] == "queued" and job["files"] is None
    assert_problem(client.get(f"/v1/jobs/{job['job_id']}/files/report.json"), 409, "CONFLICT")
    assert client.post(f"/v1/jobs/{job['job_id']}/cancel").json()["status"] == "cancelled"


def test_same_upload_cannot_be_submitted_twice(client):
    uid = upload(client).json()["upload_id"]
    client.post("/v1/jobs", json={"upload_id": uid, "line": LINE})
    assert_problem(client.post("/v1/jobs", json={"upload_id": uid, "line": LINE}), 409, "CONFLICT")


def test_queue_full_is_429_and_retryable(client):
    for _ in range(2):
        client.post("/v1/jobs", json={"upload_id": upload(client).json()["upload_id"], "line": LINE})
    r = client.post("/v1/jobs", json={"upload_id": upload(client).json()["upload_id"], "line": LINE})
    body = assert_problem(r, 429, "QUEUE_FULL")
    assert body["retryable"] is True and r.headers["retry-after"]


def test_invalid_line_rejected(client):
    uid = upload(client).json()["upload_id"]
    short = {**LINE, "bx": 0.301}
    assert_problem(client.post("/v1/jobs", json={"upload_id": uid, "line": short}), 400, "INPUT_INVALID_LINE")
    assert_problem(client.post("/v1/jobs", json={"upload_id": uid, "line": {**LINE, "ax": 2}}), 422, "INPUT_INVALID")


def test_unknown_ids_and_path_tricks_are_404(client):
    assert_problem(client.get("/v1/jobs/0123456789abcdef"), 404, "NOT_FOUND")
    assert_problem(client.get("/v1/jobs/../../etc/passwd"), 404, "NOT_FOUND")
    assert_problem(client.get("/v1/jobs/0123456789abcdef/files/..%2f..%2fbagcounter.sqlite3"), 404, "NOT_FOUND")


def test_readyz_is_503_without_a_worker(client):
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["checks"]["worker"]["ok"] is False and r.json()["checks"]["database"]["ok"] is True


def test_camera_password_is_never_returned(client):
    r = client.post("/v1/cameras", json={"name": "cam", "url": "rtsp://admin:s3cret@10.0.0.5:554/live", "line": LINE})
    assert r.status_code == 201
    assert "s3cret" not in r.text and "admin:***@10.0.0.5:554" in r.text
    assert "s3cret" not in client.get("/v1/cameras").text


def test_camera_url_scheme_validated(client):
    assert_problem(client.post("/v1/cameras", json={"name": "c", "url": "file:///etc/passwd"}), 422, "INPUT_INVALID")


def test_camera_limit(tmp_path):
    s = Settings(data_dir=tmp_path / "d", max_cameras=1)
    with TestClient(create_app(s)) as c:
        assert c.post("/v1/cameras", json={"name": "a", "url": "rtsp://10.0.0.1/a"}).status_code == 201
        assert_problem(c.post("/v1/cameras", json={"name": "b", "url": "rtsp://10.0.0.1/b"}), 409, "CAMERA_LIMIT")

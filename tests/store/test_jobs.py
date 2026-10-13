import pytest

from bagcounter.errors import QueueFullError
from bagcounter.store import MAX_JOB_ATTEMPTS, Store

LINE = {"ax": 0.1, "ay": 0.5, "bx": 0.9, "by": 0.5, "direction": "positive"}


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "db.sqlite3")


def queued_job(store, uid="u1"):
    store.add_upload(uid, "v.mp4", 10, {})
    return store.create_job(uid, LINE, True, max_queued=10)


def test_claim_is_exclusive(store):
    queued_job(store)
    assert store.claim_job() is not None
    assert store.claim_job() is None


def test_interrupted_job_is_retried_once_then_quarantined(store):
    job = queued_job(store)
    for attempt in range(1, MAX_JOB_ATTEMPTS + 1):
        claimed = store.claim_job()
        assert claimed.id == job.id and claimed.attempts == attempt
        requeued, quarantined = store.recover_interrupted()
    assert (requeued, quarantined) == (0, 1)
    j = store.get_job(job.id)
    assert j.status == "failed" and j.error_code == "WORKER_CRASHED"
    assert store.claim_job() is None


def test_queue_bound(store):
    for i in range(3):
        store.add_upload(f"u{i}", "v.mp4", 1, {})
    store.create_job("u0", LINE, True, max_queued=2)
    store.create_job("u1", LINE, True, max_queued=2)
    with pytest.raises(QueueFullError):
        store.create_job("u2", LINE, True, max_queued=2)


def test_upload_of_active_job_is_not_expired(store):
    queued_job(store)
    assert store.expired_uploads(older_than=1e12) == []

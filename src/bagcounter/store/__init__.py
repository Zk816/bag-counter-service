from bagcounter.store.cameras import Camera, CameraStore
from bagcounter.store.db import new_id
from bagcounter.store.jobs import MAX_JOB_ATTEMPTS, Job, JobStore

__all__ = ["MAX_JOB_ATTEMPTS", "Camera", "Job", "Store", "new_id"]


class Store(JobStore, CameraStore):
    pass

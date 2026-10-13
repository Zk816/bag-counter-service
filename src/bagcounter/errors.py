from __future__ import annotations


class AppError(Exception):
    code: str = "INTERNAL"
    http_status: int = 500
    retryable: bool = False

    def __init__(self, message: str, *, context: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}


class InvalidInputError(AppError):
    code, http_status = "INPUT_INVALID", 400


class InvalidVideoError(InvalidInputError):
    code = "INPUT_INVALID_VIDEO"


class InvalidLineError(InvalidInputError):
    code = "INPUT_INVALID_LINE"


class PayloadTooLargeError(AppError):
    code, http_status = "PAYLOAD_TOO_LARGE", 413


class NotFoundError(AppError):
    code, http_status = "NOT_FOUND", 404


class ConflictError(AppError):
    code, http_status = "CONFLICT", 409


class QueueFullError(AppError):
    code, http_status, retryable = "QUEUE_FULL", 429, True


class CameraLimitError(AppError):
    code, http_status = "CAMERA_LIMIT", 409


class DiskPressureError(AppError):
    code, http_status, retryable = "DISK_PRESSURE", 503, True


class ModelError(AppError):
    code, http_status, retryable = "MODEL_NOT_READY", 503, True


class JobTimeoutError(AppError):
    code, http_status = "JOB_TIMEOUT", 504


class StorageError(AppError):
    code, http_status, retryable = "STORAGE_ERROR", 503, True


class JobCancelledError(AppError):
    code, http_status = "JOB_CANCELLED", 409

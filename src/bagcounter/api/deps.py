from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from bagcounter.config import Settings
from bagcounter.detect.openvino import ModelInfo
from bagcounter.store import Store


@dataclass(frozen=True)
class AppState:
    settings: Settings
    store: Store
    model: ModelInfo
    started_at: float


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.app
    return state

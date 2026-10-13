from pathlib import Path

import pytest

from bagcounter.config import Settings

DATA = Path(__file__).parent / "data"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", infer_threads=4)

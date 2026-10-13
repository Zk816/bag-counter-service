import json
import shutil

import pytest

from bagcounter.config import Settings
from bagcounter.errors import ModelError
from bagcounter.preflight import run_preflight, selftest


def test_preflight_refuses_a_corrupted_model(tmp_path):
    s = Settings(data_dir=tmp_path / "data")
    bad = tmp_path / "model"
    shutil.copytree(s.model_dir, bad)
    with (bad / "model.bin").open("r+b") as f:
        f.seek(1000)
        f.write(b"\xff\xff\xff\xff")
    res = run_preflight(Settings(data_dir=tmp_path / "data", model_dir=bad))
    assert not res.ok
    failed = [c for c in res.checks if not c.ok]
    assert failed[0].name == "model_files" and "checksum" in failed[0].detail
    assert res.detector is None


def test_preflight_passes_on_the_packaged_model(tmp_path):
    res = run_preflight(Settings(data_dir=tmp_path / "data", infer_threads=4))
    assert res.ok, [c for c in res.checks if not c.ok]


def test_selftest_detects_wrong_output(tmp_path):
    s = Settings(data_dir=tmp_path / "data", infer_threads=4)
    res = run_preflight(s)
    det = res.detector
    det.info.meta["selftest"] = {**det.info.meta["selftest"], "boxes": [[0, 0, 50, 50, 0.9]]}
    with pytest.raises(ModelError):
        selftest(det, s)
    assert json.dumps(det.info.meta)

"""Suite-wide fixtures.

`DEFAULT_TARGET_FEATURE` in config/const_preprocessing.py is the one switch between the binary run
(`trainDisrupted`), the 4-class run (`delayClass`) and the numeric run, and it is changed on
purpose before a grid (the cluster reads it from `main`). Most tests pin the binary behaviour and
call the trainers without choosing a target, so they must not follow that switch: this fixture
pins the training pipeline's target to `trainDisrupted` for every test. A test that needs another
target patches `src.training_pipeline.DEFAULT_TARGET_FEATURE` itself, inside its own `with` block,
which wins over this fixture (tests/test_multiclass_target.py, tests/test_regression_target.py).
"""
import pytest


@pytest.fixture(autouse=True)
def _binary_target_by_default(monkeypatch):
    import src.training_pipeline as training_pipeline
    monkeypatch.setattr(training_pipeline, "DEFAULT_TARGET_FEATURE", "trainDisrupted")

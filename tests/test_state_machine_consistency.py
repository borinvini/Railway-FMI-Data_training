"""Every preprocessing switch must have a block, and every block a switch.

filter_strong_weather_causes sat in the state machine (off) for months while
being unable to run: it required a `train_id` column that the data does not
have (the raw files carry `trainNumber`), it called `datetime.now()` on the
module (`import datetime`), and it returned an empty frame when a column was
missing, so enabling it would have emptied every month. It also selected rows
by a cause code assigned after the fact, which is selection on the outcome.
It was deleted; these tests keep the state machine, the orchestrator and the
folder constants from drifting apart again.
"""
import inspect
import re
from pathlib import Path

import config.const_preprocessing as const_preprocessing
from config.const_preprocessing import PREPROCESSING_STATE_MACHINE
from src.preprocessing_pipeline import PreprocessingPipeline

ORCHESTRATOR = inspect.getsource(PreprocessingPipeline.execute_preprocessing_pipeline_steps)
SOURCE = (Path(__file__).resolve().parent.parent / "src" / "preprocessing_pipeline.py").read_text(encoding="utf-8")


def test_the_unusable_stage_is_gone():
    assert not hasattr(PreprocessingPipeline, "filter_strong_weather_causes")
    assert "filter_strong_weather_causes" not in PREPROCESSING_STATE_MACHINE
    assert not hasattr(const_preprocessing, "FOLDER_FILTER_STRONG_WEATHER_CAUSES")


def test_every_switch_has_an_orchestrator_block():
    wired = set(re.findall(r'state_machine\.get\("([a-z_0-9]+)"', ORCHESTRATOR))
    assert set(PREPROCESSING_STATE_MACHINE) - wired == set()


def test_every_orchestrator_block_has_a_switch():
    wired = set(re.findall(r'state_machine\.get\("([a-z_0-9]+)"', ORCHESTRATOR))
    assert wired - set(PREPROCESSING_STATE_MACHINE) == set()


def test_every_stage_the_orchestrator_calls_exists():
    called = set(re.findall(r"self\.([a-z_0-9]+)\(", ORCHESTRATOR))
    missing = sorted(name for name in called if not hasattr(PreprocessingPipeline, name))
    assert missing == []


def test_every_folder_constant_is_used_by_the_pipeline():
    folders = [name for name in vars(const_preprocessing)
               if name.startswith("FOLDER_") and isinstance(getattr(const_preprocessing, name), str)]
    unused = sorted(name for name in folders if SOURCE.count(name) < 2)  # import + one use
    assert unused == []

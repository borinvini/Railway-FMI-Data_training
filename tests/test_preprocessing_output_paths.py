"""Preprocessing outputs must land under project_root (--data-root), not the CWD.

PreprocessingPipeline resolved project_root = DATA_ROOT or repo root, but stages
2- ... 19- passed the bare relative FOLDER_* constants to
save_dataframe_to_parquet and the 101- stage wrote to the bare
TRAINING_READY_OUTPUT_FOLDER. Training reads project_root/101-..., so with
RAILWAY_DATA_ROOT set (or any CWD other than the repo root) preprocessing wrote
one tree and training looked in another. Training itself already joins
project_root everywhere.
"""
import ast
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from config.const_preprocessing import (
    FOLDER_REMOVE_DUPLICATES,
    TRAINING_READY_OUTPUT_FOLDER,
)
from src.preprocessing_pipeline import PreprocessingPipeline

SOURCE = Path(__file__).resolve().parent.parent / "src" / "preprocessing_pipeline.py"


@contextmanager
def _null_logger(*args, **kwargs):
    yield MagicMock()


def _pipeline(root, log_dir):
    pipeline = PreprocessingPipeline.__new__(PreprocessingPipeline)
    pipeline.project_root = str(root)
    pipeline.training_ready_dir = os.path.join(str(root), TRAINING_READY_OUTPUT_FOLDER)
    pipeline.log_dir = str(log_dir)
    pipeline.current_file_id = "2023_01"
    return pipeline


def _frame():
    return pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})


def test_training_ready_parquet_lands_under_project_root(tmp_path, monkeypatch):
    root, cwd = tmp_path / "root", tmp_path / "cwd"
    root.mkdir()
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    pipeline = _pipeline(root, tmp_path)

    assert pipeline.save_training_ready_parquet("2023_01", _frame()) is True

    expected = root / TRAINING_READY_OUTPUT_FOLDER / "training_ready_2023_01.parquet"
    assert expected.exists()
    assert list(cwd.iterdir()) == []


def test_intermediate_saves_resolve_against_project_root(tmp_path, monkeypatch):
    root, cwd = tmp_path / "root", tmp_path / "cwd"
    root.mkdir()
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    pipeline = _pipeline(root, tmp_path)

    with patch.object(pipeline, "get_logger", _null_logger), \
         patch("src.preprocessing_pipeline.SAVE_STAGE_OUTPUTS", {"remove_duplicates"}):
        out = pipeline.remove_duplicates(dataframe=_frame(), month_id="2023_01")

    assert out is not None
    assert list((root / FOLDER_REMOVE_DUPLICATES).glob("*.parquet"))
    assert list(cwd.iterdir()) == []


def test_no_save_uses_a_bare_folder_constant():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    bare = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if name != "save_dataframe_to_parquet":
            continue
        for kw in node.keywords:
            if kw.arg == "folder_path" and isinstance(kw.value, ast.Name):
                bare.append((node.lineno, kw.value.id))
    assert bare == []

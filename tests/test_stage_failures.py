"""A stage that fails must fail the month; it must not hand the input frame back.

Stages 9, 10, 11 and 13-16 caught their own exception and returned the input
frame, so the orchestrator printed a check mark and carried on with a frame
that lacked the columns the stage should have added. Several stages (3, 9,
13-15, 17) also skipped silently when their input column was missing. With the
merge keeping the column intersection (SCHEMA_MISMATCH_STRATEGY='intersect')
one such month removes the column from all 96 months. A failed month also
left its training_ready file from an earlier run on disk for the merge to pick
up, and main.py exited 0 after printing a warning.
"""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import main as main_module
from src.preprocessing_pipeline import PreprocessingPipeline


@contextmanager
def _null_logger(*args, **kwargs):
    yield MagicMock()


@contextmanager
def _boom_logger(*args, **kwargs):
    logger = MagicMock()
    logger.info.side_effect = RuntimeError("boom")
    yield logger


def _pipeline(tmp_path):
    pipeline = PreprocessingPipeline.__new__(PreprocessingPipeline)
    pipeline.project_root = str(tmp_path)
    pipeline.log_dir = str(tmp_path)
    pipeline.current_file_id = "2023_01"
    pipeline.training_ready_dir = str(tmp_path / "training_ready")
    pipeline.preprocessed_dir = str(tmp_path / "preprocessed")
    return pipeline


@pytest.fixture(autouse=True)
def _no_parquet_writes():
    with patch("src.preprocessing_pipeline.save_dataframe_to_parquet",
               return_value="/tmp/fake.parquet"):
        yield


# stage name -> (valid minimal frame, keyword arguments)
VALID = {
    "process_actual_time_column": pd.DataFrame({"scheduledTime": ["2023-01-15T10:00:00.000Z"]}),
    "filter_columns": pd.DataFrame({"trainDelayed": [1], "Air temperature": [1.0]}),
    "convert_boolean_to_numeric": pd.DataFrame({"trainStopping": [True]}),
    "convert_hour_to_sincos": pd.DataFrame({"hour": ["10:30"]}),
    "convert_month_to_sincos": pd.DataFrame({"month": [3]}),
    "convert_dayofweek_to_sincos": pd.DataFrame({"day_of_week": [2]}),
    "convert_dayofmonth_to_sincos": pd.DataFrame({"day_of_month": [15]}),
    "drop_original_temporal_columns": pd.DataFrame({"month": [3], "hour": ["10:30"], "day_of_week": [2]}),
}


@pytest.mark.parametrize("stage", sorted(VALID))
def test_an_exception_inside_a_stage_propagates(stage, tmp_path):
    pipeline = _pipeline(tmp_path)
    with patch.object(pipeline, "get_logger", _boom_logger):
        with pytest.raises(RuntimeError, match="boom"):
            getattr(pipeline, stage)(dataframe=VALID[stage].copy(), month_id="2023_01")


@pytest.mark.parametrize("stage", sorted(VALID))
def test_a_valid_frame_still_passes(stage, tmp_path):
    pipeline = _pipeline(tmp_path)
    with patch.object(pipeline, "get_logger", _null_logger):
        out = getattr(pipeline, stage)(dataframe=VALID[stage].copy(), month_id="2023_01")
    assert out is not None and len(out) == 1


MISSING_INPUT = [
    ("process_causes_column", "causes", pd.DataFrame({"a": [1]})),
    ("process_actual_time_column", "scheduledTime", pd.DataFrame({"a": [1]})),
    ("convert_hour_to_sincos", "hour", pd.DataFrame({"a": [1]})),
    ("convert_month_to_sincos", "month", pd.DataFrame({"a": [1]})),
    ("convert_dayofweek_to_sincos", "day_of_week", pd.DataFrame({"a": [1]})),
    ("convert_dayofmonth_to_sincos", "day_of_month", pd.DataFrame({"a": [1]})),
]


@pytest.mark.parametrize("stage,column,frame", MISSING_INPUT)
def test_a_missing_required_column_fails_and_names_it(stage, column, frame, tmp_path):
    pipeline = _pipeline(tmp_path)
    kwargs = {} if stage == "process_causes_column" else {"month_id": "2023_01"}
    with patch.object(pipeline, "get_logger", _null_logger):
        with pytest.raises(ValueError) as err:
            getattr(pipeline, stage)(dataframe=frame, **kwargs)
    assert column in str(err.value)


@pytest.mark.parametrize("stage,column", [
    ("convert_hour_to_sincos", "hour"),
    ("convert_month_to_sincos", "month"),
    ("convert_dayofweek_to_sincos", "day_of_week"),
    ("convert_dayofmonth_to_sincos", "day_of_month"),
])
def test_an_all_missing_input_column_fails(stage, column, tmp_path):
    pipeline = _pipeline(tmp_path)
    frame = pd.DataFrame({column: [np.nan, np.nan]})
    with patch.object(pipeline, "get_logger", _null_logger):
        with pytest.raises(ValueError) as err:
            getattr(pipeline, stage)(dataframe=frame, month_id="2023_01")
    assert column in str(err.value)


@pytest.mark.parametrize("target", [None, "not_a_target", "differenceInMinutes"])
def test_select_target_fails_when_the_target_is_unusable(target, tmp_path):
    pipeline = _pipeline(tmp_path)
    frame = pd.DataFrame({"trainDelayed": [1], "cancelled": [0]})
    with pytest.raises(ValueError):
        pipeline.select_target_feature(dataframe=frame, target_feature=target)


def test_select_target_still_keeps_the_chosen_target(tmp_path):
    pipeline = _pipeline(tmp_path)
    frame = pd.DataFrame({"trainDelayed": [1], "cancelled": [0]})
    out = pipeline.select_target_feature(dataframe=frame, target_feature="trainDelayed")
    assert list(out.columns) == ["trainDelayed"]


@pytest.mark.parametrize("stage", [
    "process_causes_column", "process_actual_time_column", "convert_hour_to_sincos",
    "convert_month_to_sincos", "convert_dayofweek_to_sincos", "convert_dayofmonth_to_sincos",
    "drop_original_temporal_columns", "filter_columns", "convert_boolean_to_numeric",
])
def test_an_empty_frame_is_returned_unchanged(stage, tmp_path):
    pipeline = _pipeline(tmp_path)
    empty = pd.DataFrame({"a": []})
    kwargs = {} if stage == "process_causes_column" else {"month_id": "2023_01"}
    with patch.object(pipeline, "get_logger", _null_logger):
        out = getattr(pipeline, stage)(dataframe=empty, **kwargs)
    assert out is not None and out.empty


def test_nothing_to_drop_is_not_a_failure(tmp_path):
    pipeline = _pipeline(tmp_path)
    frame = pd.DataFrame({"a": [1]})
    with patch.object(pipeline, "get_logger", _null_logger):
        out = pipeline.drop_original_temporal_columns(dataframe=frame, month_id="2023_01")
    assert list(out.columns) == ["a"]


def test_an_absent_boolean_column_is_only_a_warning(tmp_path):
    pipeline = _pipeline(tmp_path)
    frame = pd.DataFrame({"a": [1]})
    with patch.object(pipeline, "get_logger", _null_logger):
        out = pipeline.convert_boolean_to_numeric(dataframe=frame, month_id="2023_01")
    assert list(out.columns) == ["a"]


def test_the_orchestrator_records_the_failure_with_its_message(tmp_path):
    pipeline = _pipeline(tmp_path)
    frame = VALID["filter_columns"].copy()
    with patch("src.preprocessing_pipeline.pd.read_parquet", return_value=frame), \
         patch.object(pipeline, "get_logger", _boom_logger):
        result = pipeline.execute_preprocessing_pipeline_steps(
            input_file_path="x.parquet", file_id="2023_01", year="2023",
            state_machine={"filter_columns": True})
    assert result["success"] is False
    assert result["steps_executed"] == []
    assert result["errors"] == ["filter_columns failed: boom"]


def test_a_failed_month_removes_its_stale_outputs(tmp_path):
    pipeline = _pipeline(tmp_path)
    for folder, name in (("training_ready", "training_ready_2023_01.parquet"),
                         ("preprocessed", "preprocessed_data_2023_01.parquet")):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / name).write_bytes(b"stale")
    keep = tmp_path / "training_ready" / "training_ready_2023_02.parquet"
    keep.write_bytes(b"other month")
    failure = {"success": False, "steps_executed": [], "errors": ["x failed: boom"],
               "file_info": {"rows": 0, "columns": 0}}
    pipeline.execute_preprocessing_pipeline_steps = MagicMock(return_value=failure)
    with patch("src.preprocessing_pipeline.PREPROCESSING_STATE_MACHINE", {"filter_columns": True}):
        summary = pipeline.run_pipeline(
            [str(tmp_path / "matched_data_flat_2023_01.parquet")])
    assert summary["failed_files"] == 1
    assert summary["successful_preprocessing"] == 0
    assert not (tmp_path / "training_ready" / "training_ready_2023_01.parquet").exists()
    assert not (tmp_path / "preprocessed" / "preprocessed_data_2023_01.parquet").exists()
    assert keep.exists()


@pytest.mark.parametrize("results,expected", [
    ({"total_files": 3, "successful_preprocessing": 3, "failed_files": 0}, 0),
    ({"total_files": 3, "successful_preprocessing": 2, "failed_files": 1}, 1),
    ({"total_files": 3, "successful_preprocessing": 0, "failed_files": 3, "error": "x"}, 1),
    ({"total_files": 3, "failed_files": 0, "error": "decision failed"}, 1),
    (None, 1),
    ({}, 1),
])
def test_preprocessing_exit_code(results, expected):
    assert main_module.preprocessing_exit_code(results) == expected

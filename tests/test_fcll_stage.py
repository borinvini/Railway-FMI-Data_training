"""The stage must stream. Concatenating 96 monthly files at ~518k rows each is
roughly 50M rows and will OOM the 64G allocation."""
import json

import pandas as pd
import pytest

from src.training_pipeline import TrainingPipeline


def _write_months(input_dir):
    for month in ("2018_01", "2018_02"):
        rows = [
            {"trainNumber": 1, "departureDate": f"{month}-01",
             "stationShortCode": st, "type": "ARRIVAL",
             "scheduledTime": f"2018-01-01T10:{i*5:02d}:00.000Z",
             "actualTime": f"2018-01-01T10:{i*5:02d}:00.000Z",
             "differenceInMinutes": 0.0, "trainStopping": True,
             "commercialStop": True, "trainDelayed": 0,
             "Air temperature": 1.0}
            for i, st in enumerate(["A", "B", "OL"])
        ]
        pd.DataFrame(rows).to_parquet(input_dir / f"matched_data_flat_{month}.parquet")


def test_the_stage_writes_one_shard_per_input_file(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_months(input_dir)

    out_dir = tmp_path / "506"
    pipeline = TrainingPipeline()
    result = pipeline.build_fcll_sequences(
        input_dir=str(input_dir), output_dir=str(out_dir),
        nf_columns=["Air temperature"], target_station="OL",
    )
    assert result["success"], result.get("error")
    assert len(list(out_dir.glob("*.npz"))) == 2, "one shard per month, streamed"
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["h"] == 5 and manifest["z"] == 5
    assert manifest["n_samples"] == 2


def test_the_stage_never_loads_all_months_at_once(tmp_path, monkeypatch):
    """Guards the streaming contract directly: pd.concat over the input glob
    is the exact mistake this stage must not make."""
    import src.training_pipeline as tp

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_months(input_dir)

    calls = []
    real_concat = pd.concat

    def spy(objs, **kwargs):
        objs = list(objs)
        calls.append(len(objs))
        return real_concat(objs, **kwargs)

    monkeypatch.setattr(tp.pd, "concat", spy)
    pipeline = TrainingPipeline()
    result = pipeline.build_fcll_sequences(
        input_dir=str(input_dir), output_dir=str(tmp_path / "506"),
        nf_columns=["Air temperature"], target_station="OL",
    )
    assert result["success"], result.get("error")
    assert all(n <= 2 for n in calls), f"concat saw {max(calls)} frames at once"


def test_a_missing_input_directory_is_an_error_not_an_empty_manifest(tmp_path):
    pipeline = TrainingPipeline()
    result = pipeline.build_fcll_sequences(
        input_dir=str(tmp_path / "nothing"), output_dir=str(tmp_path / "506"),
        nf_columns=[], target_station="OL",
    )
    assert result["success"] is False
    assert "no input parquet" in result["error"]

"""One 80/20 day split for every model, validated by day-grouped CV, with the last
year never seen.

Three bugs lived in the old 55/20/25 design:

1. split_dataset wrote a validation file (20% of the days) that no tabular trainer
   read, so a fifth of the data was simply unused.
2. The trainers tuned with a row-wise StratifiedKFold. Rows of one operating day share
   weather, disruptions and knock-on delay, so they sat in both the fitting and the
   validation fold and the CV score was optimistic. SMOTE-Tomek made it worse: it ran
   on the whole train split before the CV, so synthetic rows built from validation-fold
   neighbours were trained on.
3. fcll_net split all days of all years, 2025 included, so it had no unseen year.

Now: the newest year is held out; 2018-2024 is split 80/20 by day; the tabular models
validate with StratifiedGroupKFold over the operating days of the train split (the days
travel in a sidecar next to the train file), with class weights instead of SMOTE;
fcll_net uses the same hold-out year and the same test days, and cuts its early-stopping
days out of the train days only.
"""
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from config.const_preprocessing import GROUP_KEY_COLUMN
from src.cv_utils import groups_path, load_train_groups, make_cv, neg_pos_ratio
from src.day_split import split_days
from src.training_pipeline import TrainingPipeline

TARGET = "trainDisrupted"


def _pipeline(tmp_path):
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline.project_root = str(tmp_path)
    pipeline.script_dir = str(tmp_path)
    pipeline.log_dir = str(tmp_path / "log")
    os.makedirs(pipeline.log_dir, exist_ok=True)
    return pipeline


def _frame(n_days=40, rows_per_day=6):
    """Each row has a unique `f1`; the target rate is about 30%."""
    rng = np.random.default_rng(1)
    days = pd.date_range("2023-01-01", periods=n_days).strftime("%Y-%m-%d")
    n = n_days * rows_per_day
    return pd.DataFrame({
        GROUP_KEY_COLUMN: np.repeat(days, rows_per_day),
        "f1": np.arange(n, dtype=float),
        "f2": rng.normal(size=n),
        TARGET: (rng.random(n) < 0.3).astype(int),
    })


# --- the split ---------------------------------------------------------------

def test_split_days_is_deterministic_and_disjoint():
    days = [f"2023-01-{d:02d}" for d in range(1, 29)]
    a = split_days(days, seed=42)
    b = split_days(days, seed=42)
    assert a == b
    assert a[0] & a[1] == set()
    assert a[0] | a[1] == set(days)
    assert abs(len(a[0]) / len(days) - 0.8) < 0.05


def test_write_splits_writes_a_row_aligned_day_sidecar(tmp_path):
    df = _frame()
    split = _pipeline(tmp_path).write_splits(
        df, output_dir=str(tmp_path / "out"), identifier="t", basename="merged_data_x")
    train_path = split["files"]["train"]["path"]
    train = pd.read_parquet(train_path)
    groups = load_train_groups(train_path, len(train))
    assert groups is not None
    # f1 is a unique row id, so it recovers the original day of every train row
    expected = df.set_index("f1").loc[train["f1"], GROUP_KEY_COLUMN].to_numpy()
    assert list(groups) == list(expected)
    assert "val" not in split["files"]
    assert not os.path.exists(str(tmp_path / "out" / "merged_data_x_val.parquet"))


def test_a_second_split_run_does_not_mistake_the_sidecar_for_a_source(tmp_path):
    """The sidecar sits in the same folder as the source in this layout; a run that
    counted it would abort with 'expected exactly one parquet file'."""
    src = tmp_path / "data"
    src.mkdir()
    _frame().to_parquet(src / "merged_data.parquet", index=False)
    pipeline = _pipeline(tmp_path)
    with patch("src.training_pipeline.HOLDOUT_LAST_YEAR", False):
        first = pipeline.split_dataset(data_dir=str(src), output_dir=str(src), stratify_column=TARGET)
        second = pipeline.split_dataset(data_dir=str(src), output_dir=str(src), stratify_column=TARGET)
    assert first["success"] is True
    assert second["success"] is True


def test_the_holdout_year_stays_out_of_the_80_20_split(tmp_path):
    df = _frame(n_days=60)
    df["year"] = np.where(df[GROUP_KEY_COLUMN] >= "2023-02-15", 2024, 2023)
    src = tmp_path / "data"
    src.mkdir()
    df.to_parquet(src / "merged_data.parquet", index=False)
    pipeline = _pipeline(tmp_path)
    with patch("src.training_pipeline.HOLDOUT_LAST_YEAR", True):
        result = pipeline.split_dataset(data_dir=str(src), output_dir=str(tmp_path / "out"),
                                        stratify_column=TARGET)
    assert result["holdout_year"] == 2024
    train = pd.read_parquet(tmp_path / "out" / "merged_data_train.parquet")
    test = pd.read_parquet(tmp_path / "out" / "merged_data_test.parquet")
    holdout = pd.read_parquet(tmp_path / "out" / "merged_data_holdout.parquet")
    year_of = df.set_index("f1")["year"]
    assert set(year_of.loc[train["f1"]]) == {2023}
    assert set(year_of.loc[test["f1"]]) == {2023}
    assert set(year_of.loc[holdout["f1"]]) == {2024}


def test_scaling_carries_the_sidecar_to_the_trainers_folder(tmp_path):
    from config.const_training import (
        MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER, SPLIT_DATASET_OUTPUT_FOLDER)
    pipeline = _pipeline(tmp_path)
    split_dir = tmp_path / SPLIT_DATASET_OUTPUT_FOLDER
    df = _frame()[[GROUP_KEY_COLUMN, "f1", TARGET]]
    df["Air temperature"] = np.linspace(-10, 20, len(df))
    pipeline.write_splits(df, output_dir=str(split_dir), identifier="t", basename="merged_data_x")

    assert pipeline.scale_weather_features(data_dir=str(split_dir))["success"] is True

    scaled_train = tmp_path / MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER / "merged_data_x_train.parquet"
    groups = load_train_groups(str(scaled_train), len(pd.read_parquet(scaled_train)))
    assert groups is not None and len(set(groups)) > 1


# --- the CV ------------------------------------------------------------------

def test_cv_folds_never_share_an_operating_day():
    groups = np.repeat([f"d{i}" for i in range(50)], 8)
    y = (np.arange(len(groups)) % 4 == 0).astype(int)
    cv = make_cv(True, groups, 5, 42)
    assert isinstance(cv, StratifiedGroupKFold)
    for fit_idx, val_idx in cv.split(np.zeros((len(y), 1)), y, groups):
        assert set(groups[fit_idx]).isdisjoint(groups[val_idx])


def test_regression_cv_is_grouped_too():
    groups = np.repeat([f"d{i}" for i in range(30)], 4)
    for fit_idx, val_idx in make_cv(False, groups, 5, 42).split(np.zeros((len(groups), 1)), None, groups):
        assert set(groups[fit_idx]).isdisjoint(groups[val_idx])


def test_without_groups_the_cv_falls_back_loudly(capsys):
    from sklearn.model_selection import StratifiedKFold
    cv = make_cv(True, None, 5, 42)
    assert isinstance(cv, StratifiedKFold)
    assert "row-wise CV" in capsys.readouterr().out


def test_a_sidecar_of_the_wrong_length_is_refused(tmp_path):
    """Resampling the train rows after the split (balance_classes) would misalign
    the days; training on shifted groups is worse than failing."""
    train_path = str(tmp_path / "merged_data_x_train.parquet")
    pd.DataFrame({"d": ["a", "b", "c"]}).to_parquet(groups_path(train_path), index=False)
    with pytest.raises(ValueError, match="no longer line up"):
        load_train_groups(train_path, 5)
    assert load_train_groups(str(tmp_path / "merged_data_none_train.parquet"), 5) is None


def test_neg_pos_ratio():
    assert neg_pos_ratio([0, 0, 0, 1]) == pytest.approx(3.0)
    assert neg_pos_ratio([0, 0]) == 1.0


# --- the trainers ------------------------------------------------------------

class _SearchSpy:
    """Stands in for RandomizedSearchCV: records what it was given, then stops."""
    seen = {}

    def __init__(self, **kwargs):
        _SearchSpy.seen = {"init": kwargs}

    def fit(self, X, y, **fit_kwargs):
        _SearchSpy.seen.update(X=X, y=y, fit=fit_kwargs)
        raise RuntimeError("spy: stop after capturing the search arguments")


def _prepared_folder(tmp_path):
    df = _frame()
    folder = tmp_path / "scaled"
    _pipeline(tmp_path).write_splits(df, output_dir=str(folder), identifier="t",
                                     basename="merged_data_x")
    return folder


TRAINERS = [
    ("train_xgboost_with_randomized_search_cv", "scale_pos_weight"),
    ("train_lightgbm_with_randomized_search_cv", "scale_pos_weight"),
    ("train_random_forest_with_randomized_search_cv", "class_weight"),
    ("train_logistic_regression_with_randomized_search_cv", "class_weight"),
    ("train_naive_bayes_with_randomized_search_cv", None),
]


@pytest.mark.parametrize("method,weighting", TRAINERS)
def test_trainers_tune_with_day_grouped_cv_and_class_weights(tmp_path, method, weighting):
    folder = _prepared_folder(tmp_path)
    train = pd.read_parquet(folder / "merged_data_x_train.parquet")
    pipeline = _pipeline(tmp_path)

    with patch("src.training_pipeline.RandomizedSearchCV", _SearchSpy):
        getattr(pipeline, method)(data_dir=str(folder))

    seen = _SearchSpy.seen
    groups = seen["fit"]["groups"]
    assert len(groups) == len(train)
    cv = seen["init"]["cv"]
    assert isinstance(cv, StratifiedGroupKFold)
    for fit_idx, val_idx in cv.split(seen["X"], seen["y"], groups):
        assert set(groups[fit_idx]).isdisjoint(groups[val_idx]), "a day sits in both CV folds"

    params = seen["init"]["estimator"].get_params()
    if weighting == "scale_pos_weight":
        expected = neg_pos_ratio(train[TARGET])
        assert params["scale_pos_weight"] == pytest.approx(expected)
    elif weighting == "class_weight":
        assert params["class_weight"] == "balanced"


def test_the_tabular_pipeline_does_not_resample():
    import main
    from config.const_training import TRAINING_STATE_MACHINE
    assert TRAINING_STATE_MACHINE["balance_classes"] is False
    assert "balance_classes" not in main.PREREQUISITE_STAGES


# --- fcll_net ----------------------------------------------------------------

def _fcll_data(years=(2023, 2024, 2025), days_per_year=24, per_day=8, seed=0):
    rng = np.random.default_rng(seed)
    days = [f"{y}-{m:02d}-{d:02d}" for y in years for m in (3, 4) for d in range(1, days_per_year // 2 + 1)]
    group = np.repeat(days, per_day)
    n = len(group)
    label = rng.integers(0, 2, n).astype(np.float32)
    z = np.zeros((n, 5, 9), np.float32)
    z[:, -1, 0] = label * 10.0 + rng.normal(0, 0.1, n)
    return {
        "z_seq": z, "h_seq": np.zeros((n, 5, 9), np.float32),
        "z_mask": np.ones((n, 5), np.float32), "h_mask": np.ones((n, 5), np.float32),
        "nf": np.zeros((n, 1), np.float32), "label": label, "group": group,
    }


def test_fcll_net_never_trains_validates_or_tests_on_the_newest_year():
    from src import fcll_trainer
    data = _fcll_data()
    holdout, year = fcll_trainer._holdout_mask(data["group"])
    assert year == 2025
    idx = fcll_trainer._split_indices(data["group"], holdout, seed=42)
    years = np.array([int(g[:4]) for g in data["group"]])
    for name in ("train", "val", "test"):
        assert not (years[idx[name]] == 2025).any(), name
    assert set(years[idx["holdout"]]) == {2025}
    # every dev row lands in exactly one of train / val / test
    total = idx["train"].astype(int) + idx["val"] + idx["test"]
    assert (total[~holdout] == 1).all()
    assert (total[holdout] == 0).all()


def test_fcll_net_uses_the_same_test_days_as_the_tabular_models():
    from src import fcll_trainer
    data = _fcll_data()
    holdout, _ = fcll_trainer._holdout_mask(data["group"])
    idx = fcll_trainer._split_indices(data["group"], holdout, seed=42)
    _, tabular_test_days = split_days(data["group"][~holdout], seed=42)
    assert set(data["group"][idx["test"]]) == set(tabular_test_days)


def test_fcll_net_early_stopping_days_come_from_the_train_days_only():
    from src import fcll_trainer
    data = _fcll_data()
    holdout, _ = fcll_trainer._holdout_mask(data["group"])
    idx = fcll_trainer._split_indices(data["group"], holdout, seed=42)
    train_days, test_days = split_days(data["group"][~holdout], seed=42)
    val_days = set(data["group"][idx["val"]])
    fit_days = set(data["group"][idx["train"]])
    assert val_days | fit_days == train_days
    assert val_days.isdisjoint(fit_days)
    assert (val_days | fit_days).isdisjoint(test_days)


def test_fcll_net_reports_metrics_on_the_holdout_year():
    from src.fcll_trainer import train_fcll
    result = train_fcll(_fcll_data(), epochs=5, batch_size=64, seed=42)
    metrics = result["metrics"]
    assert metrics["holdout_year"] == 2025
    assert metrics["n_holdout"] > 0
    assert metrics["holdout"]["n"] == metrics["n_holdout"]


def test_fcll_net_without_dated_groups_has_no_holdout():
    """The unit fixtures use day ids like 'd3'; those carry no year."""
    from src import fcll_trainer
    groups = np.array([f"d{i % 20}" for i in range(200)])
    mask, year = fcll_trainer._holdout_mask(groups)
    assert year is None and not mask.any()

"""Cross-validation helpers for the tabular trainers.

The trainers tune with RandomizedSearchCV on the TRAIN split. A row-wise
StratifiedKFold puts rows of the same operating day in both the fitting and the
validation fold, and one day's rows share weather, disruptions and knock-on delay,
so the CV score is optimistic and the search picks over-fitted hyperparameters.
These helpers keep whole days together.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

GROUPS_SUFFIX = "_groups"


def groups_path(train_file):
    """Sidecar written next to `<base>_train.parquet`: `<base>_train_groups.parquet`."""
    return train_file[: -len(".parquet")] + GROUPS_SUFFIX + ".parquet"


def load_train_groups(train_file, n_rows):
    """Return the operating day of each train row, or None when there is no sidecar.

    Raises when the sidecar has a different length than the train file: the rows
    were resampled or reordered after the split, so the days no longer line up.
    """
    path = groups_path(train_file)
    if not os.path.exists(path):
        return None
    groups = pd.read_parquet(path).iloc[:, 0].to_numpy()
    if len(groups) != n_rows:
        raise ValueError(
            f"{os.path.basename(path)} has {len(groups):,} rows but the train file has "
            f"{n_rows:,}: the train rows changed after the split (class resampling?), "
            f"so the operating days no longer line up."
        )
    return groups


FILLED_SUFFIX = "_filled"


def filled_path(split_file):
    """Flags next to `<base>_<split>.parquet`: `<base>_<split>_filled.parquet`.

    One bool column, row-aligned with the split file: True where the numeric target was
    filled in (REGRESSION_CANCELLED_POLICY = 'cap'), i.e. a cancelled train with no delay.
    """
    return split_file[: -len(".parquet")] + FILLED_SUFFIX + ".parquet"


def load_filled_flags(split_file, n_rows):
    """Bool array of filled rows for a split file, or None when the policy left no flags.

    Raises on a length mismatch: the split rows changed after the flags were written.
    """
    path = filled_path(split_file)
    if not os.path.exists(path):
        return None
    flags = pd.read_parquet(path).iloc[:, 0].to_numpy(dtype=bool)
    if len(flags) != n_rows:
        raise ValueError(
            f"{os.path.basename(path)} has {len(flags):,} rows but the split file has "
            f"{n_rows:,}: the rows changed after the split, so the flags no longer line up."
        )
    return flags


def make_cv(is_classification, groups, folds, seed):
    """Day-grouped CV when `groups` is known, otherwise the old row-wise CV (loudly)."""
    if groups is None:
        print("      ⚠️  No operating-day groups next to the train file: using row-wise CV, "
              "which leaks days between folds")
        return (StratifiedKFold if is_classification else KFold)(
            n_splits=folds, shuffle=True, random_state=seed)
    if is_classification:
        return StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return GroupKFold(n_splits=folds)


class NegMaeMinutes:
    """-MAE in minutes for an estimator fitted on log1p(y + y_shift).

    The numeric trainers fit on the log-shifted delay, so a plain
    `neg_mean_absolute_error` scorer measured the CV error in log units, while every
    reported metric (MAE, RMSE, R2, wMAPE) is in minutes and the best iteration was picked
    on the former. This scorer undoes the transform on both sides, so the tuning score,
    the test score and the hold-out score are the same quantity. A class (not a closure)
    so joblib can pickle it for the parallel search.
    """

    def __init__(self, y_shift):
        self.y_shift = float(y_shift)

    def __call__(self, y_true_log, y_pred_log):
        y_true = np.expm1(np.asarray(y_true_log, dtype=float)) - self.y_shift
        y_pred = np.expm1(np.asarray(y_pred_log, dtype=float)) - self.y_shift
        return -float(np.mean(np.abs(y_true - y_pred)))


def make_minutes_mae_scorer(y_shift):
    """sklearn scorer (higher is better) wrapping NegMaeMinutes."""
    return make_scorer(NegMaeMinutes(y_shift))


class NegRmseMinutes(NegMaeMinutes):
    """-RMSE in minutes for an estimator fitted on log1p(y + y_shift)."""

    def __call__(self, y_true_log, y_pred_log):
        y_true = np.expm1(np.asarray(y_true_log, dtype=float)) - self.y_shift
        y_pred = np.expm1(np.asarray(y_pred_log, dtype=float)) - self.y_shift
        return -float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


class R2Minutes(NegMaeMinutes):
    """R2 in minutes for an estimator fitted on log1p(y + y_shift)."""

    def __call__(self, y_true_log, y_pred_log):
        y_true = np.expm1(np.asarray(y_true_log, dtype=float)) - self.y_shift
        y_pred = np.expm1(np.asarray(y_pred_log, dtype=float)) - self.y_shift
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        return float(1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot) if ss_tot > 0 else 0.0


_MINUTES_SCORERS = {"mae": NegMaeMinutes, "rmse": NegRmseMinutes, "r2": R2Minutes}


_SKLEARN_SCORING = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error",
                    "r2": "r2"}


def make_minutes_scorer(y_shift, metric="mae"):
    """sklearn scorer (higher is better), scored in minutes.

    `y_shift` is the shift of a log1p(y + y_shift) target, or None when the model is fitted
    on the raw minutes (then sklearn's own scorer is already in minutes).
    `metric` is one of 'mae' (-MAE), 'rmse' (-RMSE), 'r2'.
    """
    if metric not in _MINUTES_SCORERS:
        raise ValueError(
            f"REGRESSION_SCORE_METRIC={metric!r} is not one of {sorted(_MINUTES_SCORERS)}")
    if y_shift is None:
        return _SKLEARN_SCORING[metric]
    return make_scorer(_MINUTES_SCORERS[metric](y_shift))


def to_minutes(pred, y_shift):
    """Model output in minutes: undo log1p(y + y_shift), or pass through when y_shift is None."""
    if y_shift is None:
        return pred
    return np.expm1(pred) - y_shift


def mae_baseline(y_train, y_true, mae):
    """MAE of always predicting the train median, and how much better `mae` is.

    `mae_skill` = 1 - mae / baseline: 0 means no better than the constant, 1 is perfect,
    negative is worse. Gives an R2-independent reference for a skewed delay.
    """
    y_true = np.asarray(y_true, dtype=float)
    baseline = float(np.mean(np.abs(y_true - float(np.median(y_train)))))
    skill = float(1.0 - mae / baseline) if baseline > 0 else 0.0
    return {"baseline_mae": baseline, "mae_skill": skill}


def residual_quantiles(y_true, y_pred):
    """Quantiles of (actual - predicted) in minutes: negative = predicted too late."""
    resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    qs = {"q05": 0.05, "q25": 0.25, "q50": 0.50, "q75": 0.75, "q95": 0.95}
    return {name: float(np.quantile(resid, q)) for name, q in qs.items()}


def neg_pos_ratio(y):
    """Negatives per positive, for scale_pos_weight. 1.0 when there is no positive."""
    y = np.asarray(y).astype(bool)
    pos = int(y.sum())
    return float((len(y) - pos) / pos) if pos else 1.0

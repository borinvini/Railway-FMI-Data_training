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


def neg_pos_ratio(y):
    """Negatives per positive, for scale_pos_weight. 1.0 when there is no positive."""
    y = np.asarray(y).astype(bool)
    pos = int(y.sum())
    return float((len(y) - pos) / pos) if pos else 1.0

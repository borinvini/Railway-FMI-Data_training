# scale_weather_features: routing fix + skewed-feature scaling — Design

## Context

`scale_weather_features` (`src/training_pipeline.py:2039`) is meant to fit a `RobustScaler`
on weather features from the training split and apply it to train/test, so the data is
ready for scale-sensitive models. Investigating whether enabling this stage works
correctly for the classification training approach (`trainDelayed` target,
`XGBClassifier`/`LGBMClassifier`) surfaced two problems:

1. **Routing bug (pipeline-breaking).** The stage hardcodes its input directory to
   `MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER` (`502-select_training_cols/`) and globs
   for `merged_data_*_train.parquet` / `_test.parquet` there. But `select_training_cols`
   only ever writes a single un-split file to `502/` — the actual train/test pair is
   produced later by `split_dataset` (into `503-split_dataset/`) and optionally
   `balance_classes` (into `504-balance_classes/`). In the standard pipeline order
   (select_cols → split → balance → scale), `502/` never contains `_train`/`_test`-suffixed
   files, so `scale_weather_features` returns `"No training files found to scale"` and
   aborts the run. Enabling this stage today breaks the pipeline.

2. **Training-stage routing gap.** Even if (1) is fixed, the downstream dir-routing logic
   used by `train_xgboost_with_randomized_search_cv` / `train_lightgbm_with_randomized_search_cv`
   (`src/training_pipeline.py:805-820`, `857-875`) prioritizes `balance_classes` output
   (`504/`) over `scale_weather_features` output (`505/`). So when both stages are enabled,
   the scaled data is computed but never fed to training.

Separately: the currently-scaled feature set is a mix of shapes. `RobustScaler` is
appropriate for roughly continuous weather features, but:

- `Precipitation amount`, `Precipitation intensity`, and `Snow depth` are heavily
  zero-inflated/right-skewed. `RobustScaler`'s IQR is at or near zero for these, so
  scaling degenerates to a near-no-op for the bulk of the data (sklearn falls back to
  `scale_=1.0` when IQR is 0).
- `Wind direction` is circular (0-360°); per user decision this stays on plain
  `RobustScaler` for now (no sin/cos conversion in this round).
- Boolean (`trainStopping`, `commercialStop`) and cyclical sin/cos temporal features are
  already correctly excluded from scaling — no change needed there.

The current classification/regression models (XGBoost, LightGBM) are tree-based and
scale-invariant, so none of this changes today's model metrics. The purpose of this work
is to make the stage correct and usable ahead of any future scale-sensitive model
(linear, distance-based, neural).

## Goals

- Fix `scale_weather_features` to read from wherever the real, current train/test pair
  lives, using the same balance/split/selected priority already used elsewhere.
- Fix training-stage routing so `scale_weather_features` output is used whenever scaling
  is enabled, regardless of whether `balance_classes` also ran.
- Apply `log1p` before `RobustScaler` for `Precipitation amount`, `Precipitation
  intensity`, `Snow depth`, and their rolling-window derivatives (12h/24h/72h
  min/mean/max/cumulative columns), so scaling is meaningful for these zero-inflated
  features.
- Leave `Wind direction` and all other weather features on plain `RobustScaler`,
  unchanged.
- Persist enough metadata in the saved scaler artifact that a future consumer knows
  which columns need `log1p` applied before calling `.transform`.

## Non-goals

- No sin/cos conversion of `Wind direction` in this round (explicitly declined).
- No change to which models are trained, or to `balance_classes` internals.
- No change to the "no weather features found → copy files as-is" fallback behavior.

## Design

### 1. Input directory selection for `scale_weather_features`

Add a data-dir selection block to `scale_weather_features`, mirroring the existing
training-stage routing logic (`src/training_pipeline.py:805-820`):

```
source_dir = MERGED_BALANCED_OUTPUT_FOLDER          if balance_classes ran
           else SPLIT_DATASET_OUTPUT_FOLDER          if split_dataset ran
           else MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER if filter_delay_outliers ran
                                                       and select_training_cols didn't
           else MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER
```

This requires `scale_weather_features` to receive the state machine (or an already-resolved
`data_dir`) instead of discovering `502/` unconditionally. The calling site
(`src/training_pipeline.py:389-413`) computes the resolved directory the same way the
xgboost/lightgbm call sites do, and passes it in.

### 2. Training-stage routing update

In both `train_xgboost_with_randomized_search_cv` and
`train_lightgbm_with_randomized_search_cv` call sites, change the `_data_folder`
priority so scaling wins over balancing when both are enabled:

```
_data_folder = (
    MERGED_SCALED_TRAINING_READY_OUTPUT_FOLDER   if scale_weather_features enabled
    else MERGED_BALANCED_OUTPUT_FOLDER            if balance_classes enabled
    else SPLIT_DATASET_OUTPUT_FOLDER              if split_dataset enabled
    else MERGED_OUTLIER_FILTERED_OUTPUT_FOLDER    if filter_delay_outliers enabled
                                                    and not select_training_cols
    else MERGED_SELECTED_TRAINING_READY_OUTPUT_FOLDER
)
```

This is a one-line reordering of the existing ternary chain at both call sites.

### 3. Skewed-feature handling inside `scale_weather_features`

- New constant in `config/const_preprocessing.py`:
  `SKEWED_WEATHER_FEATURES = ['Precipitation amount', 'Precipitation intensity', 'Snow depth']`.
- After computing `available_weather_features` (base weather columns present) and
  `available_window_features` (rolling-window columns present), partition them into:
  - `skewed_cols`: columns that equal one of `SKEWED_WEATHER_FEATURES`, or are window
    columns whose name starts with one of those three base names (e.g.
    `"Precipitation amount (24h cumulative)"`).
  - the remainder stays as-is.
- Before fitting/transforming, apply `np.log1p` in place to `skewed_cols` on both the
  train and test weather-feature frames (test uses the same transform, no separate
  fitting — `log1p` is deterministic, not data-dependent).
- Fit the single `RobustScaler` on the full `available_weather_features +
  available_window_features` matrix (with `log1p` already applied to `skewed_cols`),
  same as today. Transform train and test the same way.

### 4. Saved artifact format

`weather_scaler.joblib` changes from a bare `RobustScaler` to a dict:

```python
{
    "scaler": scaler,                      # fitted RobustScaler
    "weather_features": available_weather_features + available_window_features,
    "skewed_features": skewed_cols,         # subset needing log1p before scaler.transform
}
```

No current code loads this artifact outside of `scale_weather_features` itself
(verified via repo search), so this is a clean format change with no back-compat
concern.

### 5. Scaling summary updates

`scaling_summary.txt` gains a section listing which features received the `log1p`
pre-transform, in addition to the existing scaled/not-found listing.

### 6. Testing

- Extend `tests/test_xgboost_data_dir_routing.py` (or a sibling test file) with cases for:
  - `scale_weather_features` resolving to `504/`, `503/`, `502/` depending on state
    machine flags (balance/split/selected priority).
  - Training-stage routing preferring `505/` over `504/` when both `scale_weather_features`
    and `balance_classes` are enabled.
- New unit test for the `log1p` + `RobustScaler` combination on a synthetic zero-inflated
  column, confirming the transform is no longer a near-no-op (e.g. asserting non-trivial
  variance in scaled output where plain `RobustScaler` on the raw column would be close
  to identity).

## Open items resolved during brainstorming

- Tree-based models (XGBoost/LightGBM) are scale-invariant — this work does not change
  current classification/regression metrics; it prepares the stage for future
  scale-sensitive models.
- Wind direction: left as plain `RobustScaler`, no sin/cos conversion (explicit decision).
- Skewed features: `log1p` then `RobustScaler`, applied to base columns and their
  matching rolling-window derivatives.

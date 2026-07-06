# balance_classes: Target-Aware, Leakage-Safe Resampling — Design

## Problem

`balance_classes` (`src/training_pipeline.py:1095-1218`) hardcodes `target_col = "differenceInMinutes"`, completely ignoring `DEFAULT_TARGET_FEATURE` (`config/const_preprocessing.py`, currently `'trainDelayed'`), which every other training stage (`split_dataset`, `train_xgboost_with_randomized_search_cv`, `train_lightgbm_with_randomized_search_cv`) already reads as the single source of truth for which column is the model's label.

This causes four distinct problems:

1. **Silent no-op**: if `select_training_cols` drops `differenceInMinutes` (e.g. because the configured target is `trainDelayed`), `balance_classes` hits its "target not found" branch and quietly saves the train file unchanged — no error, no resampling.
2. **Leakage**: `trainDelayed` is deterministically derived from `differenceInMinutes` in preprocessing (`src/preprocessing_pipeline.py:1247`: `df['trainDelayed'] = df[TRAIN_DELAYED_TARGET_COLUMN] > TRAIN_DELAY_MINUTES`). Whichever of the pair is *not* the active target still sits in the resampling feature matrix as a near-perfect predictor of the one that is.
3. **Data loss**: `df.select_dtypes(include=[np.number])` excludes `bool` columns, so `trainDelayed`, `trainStopping`, `commercialStop` are silently dropped from the balanced output and never resampled.
4. **Unrealistic synthetic rows**: `SMOTETomek` linearly interpolates *all* numeric columns fed to it, including one-hot `weather_scenario_*` columns (producing impossible fractional "memberships") and cyclical `*_sin`/`*_cos` columns (producing points off the unit circle that decode to nonsensical times).

## Goals

- `balance_classes` derives its label from `DEFAULT_TARGET_FEATURE`, supporting both the classification flow (`target = trainDelayed`) and the regression flow (`target = differenceInMinutes` or its offset variants), consistent with how `train_xgboost`/`train_lightgbm` already determine problem type via `CLASSIFICATION_PROBLEM`/`REGRESSION_PROBLEM`.
- The `trainDelayed` ↔ `differenceInMinutes` leakage pair is broken: whichever is not the active target is dropped from the saved train file, on every code path (skipped-already-balanced, skipped-target-missing, and resampled).
- Bool columns survive resampling instead of being silently dropped.
- One-hot and cyclical columns are not linearly interpolated by the resampler.

## Non-goals

- No changes to `train_xgboost_with_randomized_search_cv` / `train_lightgbm_with_randomized_search_cv` feature-column logic. They still build `feature_columns` as "everything except the target column" — if the counterpart column is present in their input file, that's a separate leakage surface, tracked as a follow-up, not fixed here.
- `cancelled` and the `differenceInMinutes_offset*` variants get correct label derivation (via `CLASSIFICATION_PROBLEM`/`REGRESSION_PROBLEM` membership) but no counterpart-column pairing — `trainDelayed ↔ differenceInMinutes` is the only documented derived relationship, so it's the only pairing wired up.
- `RESAMPLING_METHOD == "EDITED_NEAREST_NEIGHBORS"` remains unimplemented (falls through to "not handled" as today) — out of scope.
- No new config constants — categorical/cyclical column detection is by name pattern at runtime.

## Design

### Target resolution

Replace the hardcoded `target_col = "differenceInMinutes"` with:

```python
target_col = DEFAULT_TARGET_FEATURE
```

Then branch on problem type using the existing `CLASSIFICATION_PROBLEM` / `REGRESSION_PROBLEM` lists (already imported elsewhere in `training_pipeline.py`, need to be imported into scope for this method too):

- **Classification** (`target_col in CLASSIFICATION_PROBLEM`, e.g. `trainDelayed`, `cancelled`):
  `y = df[target_col].astype(int)` — used directly, no threshold re-derivation.
- **Regression** (`target_col in REGRESSION_PROBLEM`, e.g. `differenceInMinutes`, `differenceInMinutes_offset`, `differenceInMinutes_eachStation_offset`):
  `y_bins = (df[target_col] > TRAIN_DELAY_MINUTES).astype(int)` — an *internal-only* binary label used solely to drive SMOTE's resampling ratio and minority-share reporting. The real saved label remains the continuous `target_col` value (see "Feature matrix composition" below).
- **Neither** (target misconfigured): fail with an error mirroring `train_xgboost`'s existing check (`f"Target feature '{target_col}' not recognized as classification or regression problem"`), instead of silently no-op'ing.

The existing "`target_col not in df.columns`" skip branch is kept, but now checks against the resolved `target_col` (not the old hardcoded string).

### Counterpart-column pairing (leakage fix)

Define the pairing once, near the top of the method:

```python
if target_col == "trainDelayed":
    counterpart_col = "differenceInMinutes"
elif target_col in REGRESSION_PROBLEM:
    counterpart_col = "trainDelayed"
else:
    counterpart_col = None
```

On **every** return path (skip-missing-target, skip-already-balanced, and resampled), if `counterpart_col` is present in `df.columns`, drop it before saving the train output. Report it in the result dict as `dropped_counterpart_col` (the column name, or `None` if no pairing applied / column wasn't present).

This must happen even when balancing is skipped — leakage prevention shouldn't depend on whether SMOTE actually ran.

### Feature matrix composition for resampling

- **Classification flow**: `target_col` is excluded from `X` (it's the label, not a feature — same treatment `train_xgboost` gives it). After `SMOTENC`/`TomekLinks` resampling produces `y_res`, reattach it to `df_balanced` as the `target_col` column. This means the saved label is always exactly what the resampler produced — no re-derivation from a possibly-shifted interpolated value, which closes the "SMOTE mislabeling" risk from the original report.
- **Regression flow**: `target_col` **stays in `X`** so `SMOTENC` interpolates a continuous value for synthetic rows. `y_bins` (internal, not saved) only drives the resampling ratio. The interpolated `target_col` value in `X_res` is used as-is for the saved label — there's no re-binarization step, so there's no mismatch to worry about here either (regression doesn't classify against a threshold).

### Categorical/cyclical detection (auto, by name pattern)

Before building `X`, classify columns:

- **Numeric-continuous** (interpolated normally): everything in `df.select_dtypes(include=[np.number])` that isn't one-hot or cyclical.
- **Categorical for SMOTENC** (copied from a neighbor, never interpolated):
  - One-hot: column name starts with `weather_scenario_`
  - Cyclical: column name ends with `_sin` or `_cos`
  - Bool columns: cast to `int` first (`df[col].astype(int)`), then marked categorical — this is what makes them survive resampling instead of being dropped.
- **Dropped** (as today): any remaining non-numeric column (object/string/datetime, e.g. a stray `causes` column) — reported in `dropped_non_numeric_cols` as before.

`SMOTENC(categorical_features=<indices>, random_state=SMOTE_RANDOM_STATE)` is constructed with the indices of the categorical set within `X`'s column order.

### Resampler swap

Replace:
```python
resampler = SMOTETomek(random_state=SMOTE_RANDOM_STATE)
X_res, y_res = resampler.fit_resample(X, y)
```
with a manual two-step pipeline replicating what `SMOTETomek` does internally, but NC-aware for oversampling:
```python
smote_nc = SMOTENC(categorical_features=categorical_indices, random_state=SMOTE_RANDOM_STATE)
X_over, y_over = smote_nc.fit_resample(X, y)
tomek = TomekLinks()
X_res, y_res = tomek.fit_resample(X_over, y_over)
```
This is still gated by `RESAMPLING_METHOD == "SMOTE_TOMEK"` — the config name and the branch it selects don't change, only what happens inside.

New imports: `from imblearn.over_sampling import SMOTENC`, `from imblearn.under_sampling import TomekLinks` (replacing the `SMOTETomek` import, which becomes unused in this method — check if used elsewhere before removing).

### Result dict

Unchanged shape plus one new key:

```python
{
    "success": bool,
    "rows_before": int,
    "rows_after": int,
    "minority_share_before": float | None,
    "minority_share_after": float | None,
    "resampling_method": str,          # "SMOTE_TOMEK" or "NONE"
    "skipped": bool,
    "dropped_non_numeric_cols": list[str],
    "dropped_counterpart_col": str | None,   # NEW
    "train_output_path": str,
    "test_output_path": str,
}
```

## Testing approach

Extend `tests/test_balance_classes.py` (disk-based API, `data_dir=` — unchanged from the existing after-split design) with cases for:
- Classification flow (`DEFAULT_TARGET_FEATURE = 'trainDelayed'`): `differenceInMinutes` dropped from output on skip, skip-missing, and resampled paths; `trainDelayed` survives resampling with values matching `y_res` exactly; bool columns (`trainStopping`) survive as int.
- Regression flow (`DEFAULT_TARGET_FEATURE = 'differenceInMinutes'`): `trainDelayed` dropped on all paths; `differenceInMinutes` stays continuous (not binarized) in the saved output, including for synthetic rows.
- One-hot columns (`weather_scenario_*`) in synthetic rows always sum to exactly 1 across the category group (no fractional memberships).
- Cyclical columns (`hour_sin`/`hour_cos`) in synthetic rows always come from an existing row's pair (no off-circle interpolation) — verify `sin^2 + cos^2 ≈ 1` isn't a sufficient check alone; instead assert the `(sin, cos)` pair matches some row in the pre-resample data.
- Misconfigured target (e.g. `DEFAULT_TARGET_FEATURE` not in `CLASSIFICATION_PROBLEM` or `REGRESSION_PROBLEM`) returns a clear failure instead of a silent no-op.

Tests continue to use `TrainingPipeline.__new__(TrainingPipeline)` and real `tmp_path` parquet files per existing project conventions.

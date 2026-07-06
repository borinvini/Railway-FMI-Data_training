# balance_classes Target-Aware Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `balance_classes` derive its label from `DEFAULT_TARGET_FEATURE` (supporting both the `trainDelayed` classification flow and the `differenceInMinutes` regression flow), close the leakage hole between those two columns, and stop SMOTE from producing unrealistic interpolated values for one-hot/cyclical/boolean columns.

**Architecture:** All changes are confined to the `balance_classes` method (`src/training_pipeline.py:1095-1218`) and its test file (`tests/test_balance_classes.py`). The method is rewritten in four layers: (1) target resolution via `DEFAULT_TARGET_FEATURE`/`CLASSIFICATION_PROBLEM`/`REGRESSION_PROBLEM` replacing the hardcoded `"differenceInMinutes"`, (2) counterpart-column leakage drop applied uniformly across all three return paths, (3) explicit classification-target exclusion from the feature matrix with label reattachment from the resampled result, (4) categorical/cyclical/boolean-aware resampling via `SMOTENC` + `TomekLinks` replacing `SMOTETomek`.

**Tech Stack:** Python, pandas, numpy, imbalanced-learn (`SMOTENC`, `TomekLinks`), pytest, `unittest.mock.patch`

## Global Constraints

- Tests use `TrainingPipeline.__new__(TrainingPipeline)` + manual attribute assignment — never call `__init__`
- All parquet reads/writes in tests use real `tmp_path` files — no mocking of `save_dataframe_to_parquet` or pandas I/O
- `balance_classes` must NOT return a `data` key — it stays disk-based
- Tests must not depend on the ambient value of `DEFAULT_TARGET_FEATURE` in `config/const_preprocessing.py` — every test that reaches target-resolution logic patches it explicitly via `@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "<value>")`, since `DEFAULT_TARGET_FEATURE` is imported by name into `src/training_pipeline.py`'s module namespace (`src/training_pipeline.py:71`)
- `DEFAULT_TARGET_FEATURE`, `CLASSIFICATION_PROBLEM`, `REGRESSION_PROBLEM`, `TRAIN_DELAY_MINUTES` are already imported into `src/training_pipeline.py` from `config.const_preprocessing` (lines 68-80) — no new imports needed for target resolution
- `RESAMPLING_METHOD`, `IMBALANCE_THRESHOLD`, `SMOTE_RANDOM_STATE`, `MERGED_BALANCED_OUTPUT_FOLDER` are already imported from `config.const_training` — unchanged
- `SMOTENC` requires `from imblearn.over_sampling import SMOTENC` and `TomekLinks` requires `from imblearn.under_sampling import TomekLinks`; the existing `from imblearn.combine import SMOTETomek` (`src/training_pipeline.py:23`) becomes unused after Task 4 and must be removed (verified: `SMOTETomek` is referenced nowhere else in the file)
- Empirically verified (do not re-verify): `SMOTENC` treats each categorical column independently — marginal per-column values are preserved exactly (no interpolation), but joint consistency across *related* categorical columns (e.g. a one-hot family split into separate dummy columns, or a sin/cos pair) is **not** guaranteed row-by-row. Tests must assert marginal per-column properties only, never cross-column pairing invariants.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/training_pipeline.py` | Modify | Redesign `balance_classes()` method body (lines 1095-1218) across 4 tasks; update imports (line 23) |
| `tests/test_balance_classes.py` | Modify | Update fixtures to include `trainDelayed`/categorical columns; add `DEFAULT_TARGET_FEATURE` patches; add new test coverage per task |

---

## Task 1: Target resolution via `DEFAULT_TARGET_FEATURE`

**Files:**
- Modify: `src/training_pipeline.py:1095-1218` (method body)
- Modify: `tests/test_balance_classes.py` (fixtures + existing method tests)

**Interfaces:**
- Consumes: `DEFAULT_TARGET_FEATURE: str`, `CLASSIFICATION_PROBLEM: list[str]`, `REGRESSION_PROBLEM: list[str]`, `TRAIN_DELAY_MINUTES: int` (all already imported at module level in `src/training_pipeline.py`)
- Produces: `TrainingPipeline.balance_classes(data_dir=None) -> dict` — same keys as before (`success`, `rows_before`, `rows_after`, `minority_share_before`, `minority_share_after`, `resampling_method`, `skipped`, `dropped_non_numeric_cols`, `train_output_path`, `test_output_path`), plus on failure an `"error"` key whose message is `f"Target feature '{target_col}' not recognized as classification or regression problem"` when `DEFAULT_TARGET_FEATURE` is neither in `CLASSIFICATION_PROBLEM` nor `REGRESSION_PROBLEM`

- [ ] **Step 1: Update test fixtures to include `trainDelayed` and import `TRAIN_DELAY_MINUTES`**

In `tests/test_balance_classes.py`, replace the import block and the three DataFrame-building helpers:

```python
from unittest.mock import patch

import numpy as np
import pandas as pd
import os

from src.training_pipeline import TrainingPipeline
from config.const_preprocessing import TRAIN_DELAY_MINUTES
```

```python
def _make_imbalanced_df(n_punctual=300, n_delayed=100, seed=42):
    """3:1 imbalanced DataFrame with differenceInMinutes, trainDelayed, and two numeric features."""
    rng = np.random.default_rng(seed)
    punctual = rng.uniform(-4, 5, n_punctual)
    delayed = rng.uniform(6, 60, n_delayed)
    diff = np.concatenate([punctual, delayed])
    feat_a = rng.normal(0, 1, n_punctual + n_delayed)
    feat_b = rng.normal(5, 2, n_punctual + n_delayed)
    return pd.DataFrame({
        "differenceInMinutes": diff,
        "trainDelayed": diff > TRAIN_DELAY_MINUTES,
        "feature_a": feat_a,
        "feature_b": feat_b,
    })


def _make_balanced_df(n=200, seed=42):
    """50/50 split — minority share will exceed IMBALANCE_THRESHOLD (30%)."""
    rng = np.random.default_rng(seed)
    diff = np.concatenate([rng.uniform(-4, 5, n // 2), rng.uniform(6, 60, n // 2)])
    return pd.DataFrame({
        "differenceInMinutes": diff,
        "trainDelayed": diff > TRAIN_DELAY_MINUTES,
        "feature_a": rng.normal(0, 1, n),
    })
```

```python
def _make_test_df(n=50, seed=99):
    rng = np.random.default_rng(seed)
    diff = rng.uniform(-4, 60, n)
    return pd.DataFrame({
        "differenceInMinutes": diff,
        "trainDelayed": diff > TRAIN_DELAY_MINUTES,
        "feature_a": rng.normal(0, 1, n),
    })
```

`_make_pipeline` and `_write_train_test` are unchanged.

- [ ] **Step 2: Add `@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDelayed")` to every existing method-level test, and add the misconfigured-target test**

Add this decorator directly above each of these existing `def test_...(tmp_path):` lines (no other changes to their bodies): `test_skips_when_already_balanced`, `test_smote_tomek_increases_minority_count`, `test_rows_before_and_after_in_result`, `test_non_numeric_columns_are_dropped`, `test_nan_rows_dropped_before_resampling`, `test_test_file_is_copied_to_output`, `test_result_contains_required_keys`, `test_result_has_no_data_key`.

Example for one of them:

```python
@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDelayed")
def test_skips_when_already_balanced(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_balanced_df(n=200)
    test_df = _make_test_df()
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    assert result["success"] is True
    assert result["skipped"] is True
    assert result["resampling_method"] == "NONE"
```

`test_none_data_dir_returns_failure` and `test_missing_train_file_returns_failure` are unaffected — they return before target resolution runs — leave them as-is, no decorator needed.

Add this new test directly after `test_missing_train_file_returns_failure`:

```python
@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "not_a_real_target")
def test_misconfigured_target_returns_failure(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_imbalanced_df()
    test_df = _make_test_df()
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    assert result["success"] is False
    assert "not recognized" in result["error"]
```

- [ ] **Step 3: Run tests to confirm failures**

```
pytest tests/test_balance_classes.py -v -k "misconfigured_target or skips_when or smote_tomek or rows_before or non_numeric or nan_rows or test_file or required_keys or no_data_key"
```

Expected: `test_misconfigured_target_returns_failure` FAILs (method doesn't yet check target validity — currently `"not_a_real_target"` just hits the "target not found in columns" skip branch and returns success). The other 8 tests currently PASS against old code but must keep passing after Step 4 — re-run them here only to have a documented baseline; they are not required to fail at this step.

- [ ] **Step 4: Rewrite the `balance_classes` method body in `src/training_pipeline.py`**

Replace the entire method (currently lines 1095-1218, from `def balance_classes(self, data_dir=None):` through its final `return`) with:

```python
    def balance_classes(self, data_dir=None):
        if data_dir is None:
            print("    balance_classes: data_dir is None — skipping")
            return {"success": False, "error": "data_dir is None"}

        train_files = glob.glob(os.path.join(data_dir, "*_train.parquet"))
        test_files = glob.glob(os.path.join(data_dir, "*_test.parquet"))

        if not train_files:
            msg = f"No *_train.parquet found in {data_dir}"
            print(f"    balance_classes: {msg}")
            return {"success": False, "error": msg}

        train_path = train_files[0]
        df = pd.read_parquet(train_path)
        rows_before = len(df)

        target_col = DEFAULT_TARGET_FEATURE
        is_classification = target_col in CLASSIFICATION_PROBLEM
        is_regression = target_col in REGRESSION_PROBLEM

        if not (is_classification or is_regression):
            msg = f"Target feature '{target_col}' not recognized as classification or regression problem"
            print(f"    balance_classes: {msg}")
            return {"success": False, "error": msg}

        output_folder = os.path.join(self.project_root, MERGED_BALANCED_OUTPUT_FOLDER)
        os.makedirs(output_folder, exist_ok=True)

        # Copy test file(s) unchanged regardless of whether we resample
        test_output_path = None
        if test_files:
            test_src = test_files[0]
            test_output_path = os.path.join(output_folder, os.path.basename(test_src))
            shutil.copy2(test_src, test_output_path)
            print(f"    balance_classes: Copied test file to: {test_output_path}")

        if target_col not in df.columns:
            print(f"    balance_classes: '{target_col}' not found — saving train unchanged")
            train_out = os.path.join(output_folder, os.path.basename(train_path))
            df.to_parquet(train_out, index=False)
            return {
                "success": True,
                "rows_before": rows_before,
                "rows_after": rows_before,
                "minority_share_before": None,
                "minority_share_after": None,
                "resampling_method": "NONE",
                "skipped": True,
                "dropped_non_numeric_cols": [],
                "train_output_path": train_out,
                "test_output_path": test_output_path,
            }

        if is_classification:
            y = df[target_col].astype(int)
        else:
            y = (df[target_col] > TRAIN_DELAY_MINUTES).astype(int)

        class_counts = y.value_counts()
        total = len(y)
        minority_share = int(class_counts.min()) / total * 100

        print(f"\n    balance_classes: Class balance before resampling:")
        print(f"      Punctual (≤ {TRAIN_DELAY_MINUTES} min): {int(class_counts.get(0, 0)):,} ({int(class_counts.get(0, 0)) / total * 100:.1f}%)")
        print(f"      Delayed  (> {TRAIN_DELAY_MINUTES} min): {int(class_counts.get(1, 0)):,} ({int(class_counts.get(1, 0)) / total * 100:.1f}%)")
        print(f"      Minority share: {minority_share:.1f}% (threshold: {IMBALANCE_THRESHOLD}%)")

        train_out = os.path.join(output_folder, os.path.basename(train_path))

        if minority_share >= IMBALANCE_THRESHOLD:
            print(f"    balance_classes: Balance acceptable — saving train unchanged")
            df.to_parquet(train_out, index=False)
            return {
                "success": True,
                "rows_before": rows_before,
                "rows_after": rows_before,
                "minority_share_before": minority_share,
                "minority_share_after": minority_share,
                "resampling_method": "NONE",
                "skipped": True,
                "dropped_non_numeric_cols": [],
                "train_output_path": train_out,
                "test_output_path": test_output_path,
            }

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
        if non_numeric_cols:
            print(f"    balance_classes: Dropping {len(non_numeric_cols)} non-numeric column(s): {non_numeric_cols}")

        X = df[numeric_cols].copy()
        nan_mask = X.notna().all(axis=1)
        rows_with_nan = int((~nan_mask).sum())
        if rows_with_nan > 0:
            print(f"    balance_classes: Dropping {rows_with_nan:,} rows with NaN values before resampling")
            X = X[nan_mask]
            y = y[nan_mask]

        if RESAMPLING_METHOD == "SMOTE_TOMEK":
            print(f"    balance_classes: Applying SMOTETomek (random_state={SMOTE_RANDOM_STATE})...")
            resampler = SMOTETomek(random_state=SMOTE_RANDOM_STATE)
            X_res, y_res = resampler.fit_resample(X, y)
            used_method = "SMOTE_TOMEK"
        else:
            print(f"    balance_classes: RESAMPLING_METHOD='{RESAMPLING_METHOD}' not handled — saving unchanged")
            X_res, y_res = X.values, y.values
            used_method = "NONE"

        df_balanced = pd.DataFrame(X_res, columns=numeric_cols)
        rows_after = len(df_balanced)

        y_after = pd.Series(y_res)
        counts_after = y_after.value_counts()
        minority_share_after = int(counts_after.min()) / len(y_after) * 100

        print(f"\n    balance_classes: Class balance after resampling:")
        print(f"      Punctual (≤ {TRAIN_DELAY_MINUTES} min): {int(counts_after.get(0, 0)):,} ({int(counts_after.get(0, 0)) / len(y_after) * 100:.1f}%)")
        print(f"      Delayed  (> {TRAIN_DELAY_MINUTES} min): {int(counts_after.get(1, 0)):,} ({int(counts_after.get(1, 0)) / len(y_after) * 100:.1f}%)")
        print(f"      Rows: {rows_before:,} → {rows_after:,}")

        df_balanced.to_parquet(train_out, index=False)
        print(f"      ✓ Saved balanced train to: {train_out}")

        return {
            "success": True,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "minority_share_before": minority_share,
            "minority_share_after": minority_share_after,
            "resampling_method": used_method,
            "skipped": False,
            "dropped_non_numeric_cols": non_numeric_cols,
            "train_output_path": train_out,
            "test_output_path": test_output_path,
        }
```

- [ ] **Step 5: Run tests to confirm they pass**

```
pytest tests/test_balance_classes.py -v -k "misconfigured_target or skips_when or smote_tomek or rows_before or non_numeric or nan_rows or test_file or required_keys or no_data_key or none_data_dir or missing_train_file"
```

Expected: all PASS.

- [ ] **Step 6: Run full suite**

```
pytest tests/ -v
```

Expected: all pass (state-machine tests at the bottom of `test_balance_classes.py` are untouched by this task and should still pass since they mock `balance_classes` entirely).

- [ ] **Step 7: Commit**

```bash
git add src/training_pipeline.py tests/test_balance_classes.py
git commit -m "fix: derive balance_classes label from DEFAULT_TARGET_FEATURE instead of hardcoded differenceInMinutes"
```

---

## Task 2: Counterpart-column leakage fix

**Files:**
- Modify: `src/training_pipeline.py` (method body from Task 1)
- Modify: `tests/test_balance_classes.py`

**Interfaces:**
- Consumes: `target_col: str` (resolved in Task 1)
- Produces: result dict gains `"dropped_counterpart_col": str | None` — the counterpart column name if one was dropped, else `None`. Applied on all three return paths (skip-missing-target, skip-already-balanced, resampled).

- [ ] **Step 1: Add `dropped_counterpart_col` to the required-keys test and add new leakage tests**

In `tests/test_balance_classes.py`, update the `for key in (...)` tuple inside `test_result_contains_required_keys` to include the new key:

```python
    for key in ("success", "rows_before", "rows_after",
                 "minority_share_before", "minority_share_after",
                 "resampling_method", "skipped", "dropped_non_numeric_cols",
                 "dropped_counterpart_col",
                 "train_output_path", "test_output_path"):
        assert key in result, f"Missing key: {key}"
```

Add these new tests after `test_misconfigured_target_returns_failure`:

```python
@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDelayed")
def test_counterpart_dropped_on_skip_already_balanced(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_balanced_df(n=200)
    test_df = _make_test_df()
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    assert result["dropped_counterpart_col"] == "differenceInMinutes"
    saved = pd.read_parquet(result["train_output_path"])
    assert "differenceInMinutes" not in saved.columns
    assert "trainDelayed" in saved.columns


@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDelayed")
def test_counterpart_dropped_on_skip_missing_target(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_imbalanced_df().drop(columns=["trainDelayed"])
    test_df = _make_test_df()
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    assert result["skipped"] is True
    assert result["dropped_counterpart_col"] == "differenceInMinutes"
    saved = pd.read_parquet(result["train_output_path"])
    assert "differenceInMinutes" not in saved.columns


@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDelayed")
def test_counterpart_dropped_on_resample(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_imbalanced_df(n_punctual=300, n_delayed=100)
    test_df = _make_test_df()
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    assert result["dropped_counterpart_col"] == "differenceInMinutes"
    saved = pd.read_parquet(result["train_output_path"])
    assert "differenceInMinutes" not in saved.columns


@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "differenceInMinutes")
def test_counterpart_dropped_for_regression_target(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_imbalanced_df(n_punctual=300, n_delayed=100)
    test_df = _make_test_df()
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    assert result["dropped_counterpart_col"] == "trainDelayed"
    saved = pd.read_parquet(result["train_output_path"])
    assert "trainDelayed" not in saved.columns
    assert "differenceInMinutes" in saved.columns
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_balance_classes.py -v -k "counterpart or required_keys"
```

Expected: 5 FAILs (`dropped_counterpart_col` doesn't exist yet; `differenceInMinutes`/`trainDelayed` still present in saved output).

- [ ] **Step 3: Add counterpart-column logic to `balance_classes`**

In `src/training_pipeline.py`, immediately after the `if not (is_classification or is_regression):` block (right after computing `is_classification`/`is_regression` and before the `output_folder = ...` line), insert:

```python
        if target_col == "trainDelayed":
            counterpart_col = "differenceInMinutes"
        elif is_regression:
            counterpart_col = "trainDelayed"
        else:
            counterpart_col = None
```

Immediately after the test-file-copy block (right after the `print(f"    balance_classes: Copied test file to: {test_output_path}")` line, inside the `if test_files:` block's containing scope — i.e. as the next statement after that whole `if test_files:` block), insert:

```python
        dropped_counterpart_col = None
        if counterpart_col and counterpart_col in df.columns:
            print(f"    balance_classes: Dropping counterpart column '{counterpart_col}' to prevent leakage")
            df = df.drop(columns=[counterpart_col])
            dropped_counterpart_col = counterpart_col
```

Then add `"dropped_counterpart_col": dropped_counterpart_col,` as a new line in **all three** return dicts (skip-missing-target, skip-already-balanced, and the final resampled return) — for example the skip-missing-target return becomes:

```python
            return {
                "success": True,
                "rows_before": rows_before,
                "rows_after": rows_before,
                "minority_share_before": None,
                "minority_share_after": None,
                "resampling_method": "NONE",
                "skipped": True,
                "dropped_non_numeric_cols": [],
                "dropped_counterpart_col": dropped_counterpart_col,
                "train_output_path": train_out,
                "test_output_path": test_output_path,
            }
```

Apply the same single added line to the skip-already-balanced return dict and the final resampled return dict.

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/test_balance_classes.py -v -k "counterpart or required_keys"
```

Expected: all PASS.

- [ ] **Step 5: Run full suite**

```
pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/training_pipeline.py tests/test_balance_classes.py
git commit -m "fix: drop trainDelayed/differenceInMinutes counterpart column in balance_classes to prevent leakage"
```

---

## Task 3: Exclude classification target from the feature matrix and reattach resampled labels

**Files:**
- Modify: `src/training_pipeline.py` (method body from Task 2)
- Modify: `tests/test_balance_classes.py`

**Interfaces:**
- Consumes: `is_classification: bool`, `target_col: str`, `y: pd.Series`, `y_res` (from resampler)
- Produces: for classification flows, `target_col` is never part of `X`/`numeric_cols`, and the saved `df_balanced` gets `target_col` populated from `y_res` exactly (not re-derived)

- [ ] **Step 1: Add test asserting the saved label matches the resampled result exactly**

Add after `test_counterpart_dropped_for_regression_target`:

```python
@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDelayed")
def test_classification_target_excluded_from_features_and_matches_resampled_labels(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_imbalanced_df(n_punctual=300, n_delayed=100)
    test_df = _make_test_df()
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    saved = pd.read_parquet(result["train_output_path"])
    assert "trainDelayed" in saved.columns
    assert set(saved["trainDelayed"].unique()) <= {0, 1}
    counts = saved["trainDelayed"].value_counts()
    minority_share = int(counts.min()) / len(saved) * 100
    assert abs(minority_share - result["minority_share_after"]) < 0.5
```

- [ ] **Step 2: Run test to confirm it fails**

```
pytest tests/test_balance_classes.py -v -k test_classification_target_excluded_from_features_and_matches_resampled_labels
```

Expected: FAIL — `trainDelayed` is bool-typed, so `df.select_dtypes(include=[np.number])` already excludes it from `X` today, but it is never reattached to `df_balanced` after resampling, so `"trainDelayed" in saved.columns` fails.

- [ ] **Step 3: Update the feature-matrix and post-resample logic**

In `src/training_pipeline.py`, replace this block (built in Task 1/2):

```python
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
```

with:

```python
        feature_source_cols = [c for c in df.columns if c != target_col] if is_classification else list(df.columns)
        numeric_cols = [c for c in feature_source_cols if c in df.select_dtypes(include=[np.number]).columns]
        non_numeric_cols = [c for c in feature_source_cols if c not in numeric_cols]
```

Then replace:

```python
        df_balanced = pd.DataFrame(X_res, columns=numeric_cols)
        rows_after = len(df_balanced)
```

with:

```python
        df_balanced = pd.DataFrame(X_res, columns=numeric_cols)
        if is_classification:
            df_balanced[target_col] = np.asarray(y_res)
        rows_after = len(df_balanced)
```

- [ ] **Step 4: Run test to confirm it passes**

```
pytest tests/test_balance_classes.py -v -k test_classification_target_excluded_from_features_and_matches_resampled_labels
```

Expected: PASS.

- [ ] **Step 5: Run full suite**

```
pytest tests/ -v
```

Expected: all pass — in particular, re-check `test_non_numeric_columns_are_dropped` (the injected `"causes"` object column must still land in `dropped_non_numeric_cols`) and `test_counterpart_dropped_on_resample` (unaffected, counterpart drop happens before this point).

- [ ] **Step 6: Commit**

```bash
git add src/training_pipeline.py tests/test_balance_classes.py
git commit -m "fix: exclude classification target from balance_classes feature matrix, reattach from resampled labels"
```

---

## Task 4: SMOTE-NC for categorical/cyclical/boolean columns; regression target stays continuous

**Files:**
- Modify: `src/training_pipeline.py` (imports + method body from Task 3)
- Modify: `tests/test_balance_classes.py`

**Interfaces:**
- Consumes: `from imblearn.over_sampling import SMOTENC`, `from imblearn.under_sampling import TomekLinks`
- Produces: bool columns survive resampling as `{0, 1}` int columns; `weather_scenario_*` and `*_sin`/`*_cos` columns are never linearly interpolated (every value in the output is a value that existed in the input for that column); regression-flow `target_col` stays a continuous float in the saved output (it is not excluded from `X`, so `SMOTENC` interpolates it like any other continuous numeric feature)

- [ ] **Step 1: Add a categorical-feature fixture and new tests**

Add this helper after `_make_test_df` in `tests/test_balance_classes.py`:

```python
def _make_categorical_df(n_punctual=300, n_delayed=100, seed=7):
    rng = np.random.default_rng(seed)
    punctual = rng.uniform(-4, 5, n_punctual)
    delayed = rng.uniform(6, 60, n_delayed)
    diff = np.concatenate([punctual, delayed])
    n = n_punctual + n_delayed
    hours = rng.integers(0, 24, n)
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)
    scenario_idx = rng.integers(0, 3, n)
    weather_blizzard = (scenario_idx == 0).astype(int)
    weather_clear = (scenario_idx == 1).astype(int)
    weather_rain = (scenario_idx == 2).astype(int)
    train_stopping = rng.integers(0, 2, n).astype(bool)
    return pd.DataFrame({
        "differenceInMinutes": diff,
        "trainDelayed": diff > TRAIN_DELAY_MINUTES,
        "feature_a": rng.normal(0, 1, n),
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "weather_scenario_Blizzard": weather_blizzard,
        "weather_scenario_Clear": weather_clear,
        "weather_scenario_Rain": weather_rain,
        "trainStopping": train_stopping,
    })
```

Add these tests after `test_classification_target_excluded_from_features_and_matches_resampled_labels`:

```python
@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDelayed")
def test_one_hot_columns_stay_binary(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_categorical_df()
    test_df = _make_test_df()
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    saved = pd.read_parquet(result["train_output_path"])
    for col in ("weather_scenario_Blizzard", "weather_scenario_Clear", "weather_scenario_Rain"):
        assert set(saved[col].unique()) <= {0, 1}, f"{col} was interpolated into a fractional value"


@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDelayed")
def test_cyclical_columns_are_not_interpolated(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_categorical_df()
    test_df = _make_test_df()
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    saved = pd.read_parquet(result["train_output_path"])
    existing_sin = set(np.round(train_df["hour_sin"], 6))
    existing_cos = set(np.round(train_df["hour_cos"], 6))
    saved_sin = set(np.round(saved["hour_sin"], 6))
    saved_cos = set(np.round(saved["hour_cos"], 6))
    assert saved_sin <= existing_sin, "hour_sin contains values not present in the original data (interpolated)"
    assert saved_cos <= existing_cos, "hour_cos contains values not present in the original data (interpolated)"


@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "trainDelayed")
def test_bool_columns_survive_resampling(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_categorical_df()
    test_df = _make_test_df()
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    saved = pd.read_parquet(result["train_output_path"])
    assert "trainStopping" in saved.columns
    assert set(saved["trainStopping"].unique()) <= {0, 1}


@patch("src.training_pipeline.DEFAULT_TARGET_FEATURE", "differenceInMinutes")
def test_regression_target_stays_continuous(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    train_df = _make_imbalanced_df(n_punctual=300, n_delayed=100)
    test_df = _make_test_df()
    data_dir = _write_train_test(tmp_path, train_df, test_df)
    result = pipeline.balance_classes(data_dir=data_dir)
    assert result["success"] is True
    assert result["skipped"] is False
    saved = pd.read_parquet(result["train_output_path"])
    assert saved["differenceInMinutes"].dtype.kind == "f"
    assert saved["differenceInMinutes"].nunique() > 10
```

- [ ] **Step 2: Run tests to confirm failures**

```
pytest tests/test_balance_classes.py -v -k "one_hot_columns_stay_binary or cyclical_columns_are_not_interpolated or bool_columns_survive or regression_target_stays_continuous"
```

Expected: `test_bool_columns_survive_resampling` FAILs (`trainStopping` currently dropped as non-numeric since bool is excluded from `select_dtypes(np.number)`). The other three may currently PASS by coincidence against plain `SMOTETomek` (no per-column categorical guarantee, but small feature counts can pass incidentally) — re-run after Step 3 regardless; all four must pass at Step 4.

- [ ] **Step 3: Swap the resampler for `SMOTENC` + `TomekLinks`, and mark bool/one-hot/cyclical columns as categorical**

In `src/training_pipeline.py`, update the import block. Replace:

```python
from imblearn.combine import SMOTETomek
```

with:

```python
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import TomekLinks
```

Then replace this block (from Task 3):

```python
        feature_source_cols = [c for c in df.columns if c != target_col] if is_classification else list(df.columns)
        numeric_cols = [c for c in feature_source_cols if c in df.select_dtypes(include=[np.number]).columns]
        non_numeric_cols = [c for c in feature_source_cols if c not in numeric_cols]
        if non_numeric_cols:
            print(f"    balance_classes: Dropping {len(non_numeric_cols)} non-numeric column(s): {non_numeric_cols}")

        X = df[numeric_cols].copy()
```

with:

```python
        feature_source_cols = [c for c in df.columns if c != target_col] if is_classification else list(df.columns)
        bool_cols = [c for c in feature_source_cols if df[c].dtype == bool]
        for c in bool_cols:
            df[c] = df[c].astype(int)

        numeric_cols = [c for c in feature_source_cols if c in df.select_dtypes(include=[np.number]).columns]
        non_numeric_cols = [c for c in feature_source_cols if c not in numeric_cols]
        if non_numeric_cols:
            print(f"    balance_classes: Dropping {len(non_numeric_cols)} non-numeric column(s): {non_numeric_cols}")

        X = df[numeric_cols].copy()
```

Then replace:

```python
        if RESAMPLING_METHOD == "SMOTE_TOMEK":
            print(f"    balance_classes: Applying SMOTETomek (random_state={SMOTE_RANDOM_STATE})...")
            resampler = SMOTETomek(random_state=SMOTE_RANDOM_STATE)
            X_res, y_res = resampler.fit_resample(X, y)
            used_method = "SMOTE_TOMEK"
```

with:

```python
        if RESAMPLING_METHOD == "SMOTE_TOMEK":
            categorical_cols = [
                c for c in numeric_cols
                if c in bool_cols
                or c.startswith("weather_scenario_")
                or c.endswith("_sin")
                or c.endswith("_cos")
            ]
            categorical_indices = [numeric_cols.index(c) for c in categorical_cols]

            print(f"    balance_classes: Applying SMOTENC (random_state={SMOTE_RANDOM_STATE}, categorical_features={len(categorical_indices)})...")
            smote_nc = SMOTENC(categorical_features=categorical_indices, random_state=SMOTE_RANDOM_STATE)
            X_over, y_over = smote_nc.fit_resample(X, y)

            print(f"    balance_classes: Applying TomekLinks cleaning...")
            tomek = TomekLinks()
            X_res, y_res = tomek.fit_resample(X_over, y_over)
            used_method = "SMOTE_TOMEK"
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/test_balance_classes.py -v -k "one_hot_columns_stay_binary or cyclical_columns_are_not_interpolated or bool_columns_survive or regression_target_stays_continuous"
```

Expected: all 4 PASS.

- [ ] **Step 5: Run full test suite**

```
pytest tests/ -v
```

Expected: all pass. In particular, `test_non_numeric_columns_are_dropped` must still pass (the injected `"causes"` object column is still non-numeric and still gets dropped — it's never marked categorical since categorical detection only runs over `numeric_cols`), and `test_smote_tomek_increases_minority_count` / `test_classification_target_excluded_from_features_and_matches_resampled_labels` must still pass with the new resampler.

- [ ] **Step 6: Commit**

```bash
git add src/training_pipeline.py tests/test_balance_classes.py
git commit -m "feat: use SMOTENC + TomekLinks in balance_classes to avoid interpolating one-hot/cyclical/boolean columns"
```

---

## Task 5: Full-suite verification and cross-check against `_BALANCE_SUCCESS` state-machine mock

**Files:**
- Modify: `tests/test_balance_classes.py` (state-machine mock dict only)

**Interfaces:**
- Consumes: nothing new
- Produces: nothing new — this task only brings the mocked `_BALANCE_SUCCESS` dict (used by the state-machine tests at the bottom of the file, which mock `balance_classes` entirely) up to date with the real result-dict shape, so a future reader isn't misled by a stale mock.

- [ ] **Step 1: Update `_BALANCE_SUCCESS` to include `dropped_counterpart_col`**

In `tests/test_balance_classes.py`, update:

```python
_BALANCE_SUCCESS = {
    "success": True,
    "rows_before": 320,
    "rows_after": 380,
    "minority_share_before": 25.0,
    "minority_share_after": 48.0,
    "resampling_method": "SMOTE_TOMEK",
    "skipped": False,
    "dropped_non_numeric_cols": [],
    "dropped_counterpart_col": "differenceInMinutes",
    "train_output_path": "/fake/train.parquet",
    "test_output_path": "/fake/test.parquet",
}
```

Also update the inline dict literal inside `test_split_dataset_no_longer_routes_based_on_balance` (it duplicates `_BALANCE_SUCCESS`'s shape inline) to add the same key:

```python
    mock_balance.return_value = {"success": True, "rows_before": 320, "rows_after": 380, "minority_share_before": 25.0, "minority_share_after": 48.0, "resampling_method": "SMOTE_TOMEK", "skipped": False, "dropped_non_numeric_cols": [], "dropped_counterpart_col": "differenceInMinutes", "train_output_path": "/fake/train.parquet", "test_output_path": "/fake/test.parquet"}
```

- [ ] **Step 2: Run the full test suite**

```
pytest tests/ -v
```

Expected: all tests pass — this is a pure documentation-consistency change (mocks aren't validated against the real return shape by Python), so nothing should newly fail or pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_balance_classes.py
git commit -m "test: keep balance_classes state-machine mock in sync with dropped_counterpart_col result key"
```

---

## Self-Review Notes

- **Spec coverage:** Target resolution via `DEFAULT_TARGET_FEATURE` (Task 1), counterpart-column leakage fix on all paths (Task 2), classification target exclusion + label reattachment (Task 3), regression target staying continuous (Task 4), categorical/cyclical/bool handling via `SMOTENC`+`TomekLinks` (Task 4), `dropped_counterpart_col` result key (Task 2) — all covered. `cancelled`/offset-variant targets get correct derivation via the `is_classification`/`is_regression` branch in Task 1 without special counterpart pairing, per the spec's non-goals.
- **Placeholder scan:** No TBD/TODO markers; every step shows complete code.
- **Type consistency:** `target_col`, `is_classification`, `is_regression`, `counterpart_col`, `dropped_counterpart_col`, `numeric_cols`, `bool_cols`, `categorical_cols`/`categorical_indices` are named consistently across all four tasks' code blocks.

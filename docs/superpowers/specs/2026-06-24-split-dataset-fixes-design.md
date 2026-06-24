# split_dataset — Four Correctness and Consistency Fixes

**Date:** 2026-06-24  
**Stage:** Stage 4 — `split_dataset`  
**Files touched:** `src/training_pipeline.py`, `config/const_training.py`

---

## Problems Being Solved

### Issue 1 — Fragile problem-type detection heuristic

`split_dataset` uses `df[target_column].nunique() <= 10` to decide whether to apply stratified splitting. This is inconsistent with every other stage (XGBoost, threshold optimization, etc.), which use `target_feature in CLASSIFICATION_PROBLEM`. An ordinal target with 11 classes would be silently misdetected as regression.

### Issue 2 — `split_summary.txt` saved to the wrong folder

The summary file is written to `merged_training_ready_dir` (the 502/ source folder) instead of `split_output_dir` (the 503/ output folder). The train/test parquet files land in 503/ but the summary that describes them lands in 502/.

### Issue 3 — `csv_files` parameter is accepted but never used

The method signature includes `csv_files=None`, the dispatcher passes `csv_files`, and the parameter is silently ignored — the method discovers files itself from disk. This is dead interface surface.

### Issue 4 — `random_state=42` is a magic default, not a named constant

`TEST_SIZE` is a named constant in `const_training.py` imported and used explicitly. `random_state` defaults to the magic number `42` with no config constant, making reproducibility configuration inconsistent.

---

## Design

### Issue 1 — Use `CLASSIFICATION_PROBLEM` for detection

Replace the `nunique() <= 10` heuristic with:

```python
is_classification = target_column in CLASSIFICATION_PROBLEM
```

`CLASSIFICATION_PROBLEM` is already imported at line 76 of `training_pipeline.py`. The heuristic and its inline comment are deleted. Behaviour is now identical to Stage 6 (XGBoost).

### Issue 2 — Fix summary path

Change the summary path from:

```python
summary_path = os.path.join(merged_training_ready_dir, summary_filename)
```

to:

```python
summary_path = os.path.join(split_output_dir, summary_filename)
```

`split_output_dir` (503/) is already computed at line 1761. No other changes to the summary content.

### Issue 3 — Remove `csv_files` parameter

- Remove `csv_files` from the `split_dataset` method signature.
- Remove the corresponding docstring `Parameters` entry.
- Update the dispatcher call site to not pass `csv_files`.

### Issue 4 — Add `RANDOM_STATE` constant

Add to `config/const_training.py` after `TEST_SIZE = 0.2`:

```python
RANDOM_STATE = 42
```

Add `RANDOM_STATE` to the `from config.const_training import (...)` block in `training_pipeline.py`. Change the `split_dataset` default parameter from `random_state=42` to `random_state=RANDOM_STATE`.

---

## Out of Scope

- Changes to how `split_dataset` discovers its source file.
- Changes to summary content or format.
- Any routing logic changes.

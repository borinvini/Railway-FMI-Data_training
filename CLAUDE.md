# CLAUDE.md

Python pipeline that trains train-delay models (`trainDisrupted`: delayed by more than `TRAIN_DELAY_MINUTES` (5) or cancelled, at target station `OL`, Oulu; `trainDelayed` is the pure delay label) on the matched train+weather parquet files produced by the sibling project `Railway-FMI-Data_fetcher`. It runs locally (Windows) and on CSC Roihu. Models: xgboost, lightgbm, random_forest, logistic_regression, naive_bayes (all RandomizedSearchCV) and `fcll_net` (PyTorch FCNN + 2 LSTMs over the 5 preceding trains and 5 preceding stations, after Huang et al. 2020). Other sibling folders: `Railway-FMI-Data_viewer` (Streamlit) and `Railway-FMI-Data_training` (older, non-CSC checkout of this repo).

## Running
- Conda env `venv_rail_fmi_training`. Interpreter: `C:\Users\vinic\miniconda3\envs\venv_rail_fmi_training\python.exe` (the `python` on PATH is base miniconda, not this env).
- Run: `<python> -u -X utf8 main.py [flags]`. `-X utf8` is required on Windows: the code prints `→ ✓ ✗ ⊝ ⚠️` and a cp1252 console crashes, even inside the error handler. Batch scripts export `PYTHONIOENCODING=utf-8`.
- Tests: `<python> -X utf8 -m pytest tests`, from the repo root (some tests use relative paths such as `data/input/...` and `main.py`). Two real-data tests skip when `data/input` has no parquet.
- Env files: `environment.yml` (Windows) is **UTF-16 LE with a BOM**, keep that encoding; it lacks pytest, lightgbm and pyarrow, which are installed in the env anyway. `environment-linux.yml` is the Roihu/Tykky env (LF line endings, forced by `.gitattributes`).
- CLI flags (`main.py`):
  - `--model {xgboost,lightgbm,random_forest,logistic_regression,naive_bayes,fcll_net}`: one trainer plus its prerequisites. Tabular models get `PREREQUISITE_STAGES` (500–503 and 505; `balance_classes` is off); `fcll_net` gets only `build_fcll_sequences`.
  - `--stages a,b,c`: turn on only these stages of `TRAINING_STATE_MACHINE`. Cannot be combined with `--model`. Prefer `--model`; `--stages` must cover a contiguous prefix of the chain or a stage silently reads stale folders.
  - `--data-root` (`RAILWAY_DATA_ROOT`), `--n-jobs` (`RAILWAY_N_JOBS`, -1 or positive), `--search-iterations` (`RAILWAY_SEARCH_ITERATIONS`, minimum 10: the trainer loops over `range(10, N+1, 10)`).
  - `--columns-file`, `--scenario <index|name>` (needs `--columns-file`), `--list-scenarios`, `--dump-columns` (the last two print and exit). `--dump-columns` prints names one per line, ready for `config/features.txt`.
  - With no `--columns-file` (and no `RAILWAY_COLUMNS_FILE`), `main.py` uses `config/features.txt` when it exists, found relative to `main.py`, so the IDE run button works from any directory. If it is missing and a stage needs columns, the run is refused up front (exit 1) instead of falling into the interactive prompt. There is no hard-coded column list any more.
- The flags only set `RAILWAY_*` env vars, because config constants are bound at import time. So `main.py` may import only `argparse`, `os`, `sys` at module level (`tests/test_cli.py` pins it).
- Exit code is 1 when `result["errors"]` is non-empty, even if `success` is True.
- **Which pipeline runs** is set by `EXECUTE_PREPROCESSING_DATA_PIPELINE` / `EXECUTE_TRAINING_PIPELINE` in `config/const.py`. Check them before running and before pushing: commit c77c574 pushed preprocessing=True to the cluster and array 727792 trained nothing (4bdba09 reverted it).

## Pipeline
- **Preprocessing** (`src/preprocessing_pipeline.py`, `PreprocessingPipeline`): per month, reads `data/input/matched_data_flat_YYYY_MM.parquet`, writes `data/output/<N>-<stage>/` (2- … 19-, then `100-preprocessed` and `101-preprocessed_training_ready`). Switches: `PREPROCESSING_STATE_MACHINE` in `config/const_preprocessing.py`. `convert_scheduled_time_to_local` (off by default, folder `8c-`) adds `scheduledTime_local` in `LOCAL_TIMEZONE` for the calendar stage; every timestamp is UTC already, so it only changes what the clock feature means.
- **Training** (`src/training_pipeline.py`, `execute_training_pipeline_steps`) runs in this fixed code order, and each stage returns early on error: 500-merge_data_files, 501-filter_delay_outliers, 502-select_training_cols, 503-split_dataset, 504-balance_classes, 505-scale_weather_features, 506-build_fcll_sequences, 700-shap_correlation_analysis, 1005-fcll_net, then 1000 xgboost, 1001 lightgbm, 1002 random_forest, 1003 logistic_regression, 1004 naive_bayes. Switches: `TRAINING_STATE_MACHINE` in `config/const_training.py`.
- **FCLL-Net**: `src/fcll_sequences.py` (per-visit operational features, Z/H tensors with pre-padding and masks), `src/fcll_model.py`, `src/fcll_trainer.py` (`pos_weight`, same hold-out year and 80/20 test days as the tabular models, early stopping on 15% of the train days). Stage 506 reads the raw `data/input/matched_data_flat_*.parquet` one month at a time and writes `fcll_<YYYY_MM>.npz` + `manifest.json`; stage 1005 writes `fcll_net.pt`, `metrics.json`, `val_loss_history.json`. `scripts/fcll_feasibility_check.py` measures coverage (run from the repo root).

## Config
- `config/const.py`: `DATA_ROOT` (empty = repo root), the two `EXECUTE_*` flags, `INPUT_FOLDER`, `OUTPUT_FOLDER`.
- `config/const_preprocessing.py`: `TARGET_STATION_CODE='OL'`, `TRAIN_DELAY_MINUTES=5`, `GROUP_KEY_COLUMN="departureDate"`, WAWA group maps, rolling windows.
- `config/const_training.py`: state machine, `SELECTED_COLUMNS` (empty; filled from the columns file), split fractions (80/20 by day + last-year hold-out, no validation split), `FCLL_VAL_FRACTION`, `FCLL_H_TRAINS` / `FCLL_Z_STATIONS`, `SEARCH_N_JOBS` (`RAILWAY_N_JOBS`, else `SLURM_CPUS_PER_TASK`, else -1), `MODEL_N_JOBS=1`.
- `config/columns_file.py`: parser for `--columns-file`; standard library only, to avoid an import cycle. A scenario catalogue has `# N - NAME` banner lines. `features.example.txt` and `scenarios.example.txt` are committed; `features.txt` and `scenarios.txt` are the gitignored working copies and travel to Roihu by scp (`hpc/push-features.sh`), not git.

## Data
- Input: `data/input/matched_data_flat_2018_01 … 2025_12.parquet` (96 months), copied by hand from the fetcher's `data/matched/parquet/`. The local copies predate the fetcher's rolling-order, schema-drift and 60-minute-gap fixes (rolling windows filled per column from different donor stations; `commercialTrack` text/float drift across months), so re-copy them from the fetcher before trusting a new run.
- Output: `data/output/<N>-*/`, logs in `data/output/log/`, fetched cluster results in `results/<scenario name>/`.
- `.gitignore` excludes by extension (`*.parquet *.csv *.json *.txt *.png *.pdf *.pkl *.joblib`), plus `.vscode/`, `docs/superpowers/` and `.worktrees/`. `docs/*.md` is tracked. Because `*.txt` is ignored, `features.txt` and `scenarios.txt` are untracked.
- Always re-preprocess **all** months, never a subset: `SCHEMA_MISMATCH_STRATEGY='intersect'` silently drops columns missing in any month.

## CSC Roihu
Puhti and Mahti are retired. Project `project_2019266`, ssh alias `roihu`. Code `/projappl/project_2019266/railway-fmi-code`, env `/projappl/project_2019266/railway-env` (Tykky), data `/scratch/project_2019266/railway-fmi` (250 GiB quota, untouched files deleted after 180 days). Cluster branch: `main`. Code goes by git, features by scp, data by `stage_data.sh`. Runbooks: `docs/ROIHU-QUICKSTART.md`, `docs/ROIHU-RERUN.md`, `docs/CSC-SETUP.md`.
- `hpc/roihu-auth.sh [-r]`: daily 24 h SSH certificate (a morning `Permission denied (publickey)` is expected). Uses `hpc/vendor/csc_cert.py`; do not edit it.
- `hpc/stage_data.sh [--raw]`: default sends `101-preprocessed_training_ready` (~26 MB); `--raw` also sends `data/input` (~2.6 GB), which `fcll_net` needs. Unknown options are refused.
- `hpc/push-features.sh [file]`: validates with the real parser, then scps into the cluster clone.
- `hpc/models.sh`: `MODELS=(...)` order is part of the contract (task id → model); only append.
- `hpc/submit-scenarios.sh [--scenarios N|N-M] [--dry-run]`: sizes the array from `--list-scenarios` and calls sbatch. Always use it; a bare sbatch lost scenarios 9–13 in run 581402.
- `hpc/train_scenarios.sh`: one task per scenario×model (13×6 = 78), `scenario = id/6+1`, `model = id%6`; each task gets its own root `run_sNN_<model>` with `101-…` symlinked in.
- `hpc/train_array.sh` (one model per task), `train_all.sh` (whole state machine, 48 h), `smoke_test.sh` (xgboost, 10 iterations).
- `hpc/fetch-results.sh [--models|--train-all|--scenarios]` + `sort_results.py`: pull results into `results/<scenario>/`.
- Run `git pull` on Roihu after pushing, before submitting.

## Invariants (each was a real bug; tests in `tests/` pin them)
- Split by operating day (`departureDate`), never by random rows; the key is dropped by the trainers after the split.
- Weather NaN are imputed after the split (stage 503, `write_splits`), from TRAIN rows only: the median of the row's monthly file (`year_month`, stamped by stage 500), else of its calendar month over the train years, else the global train median. A single all-year median put -6 C winter days at +2.9 C. The key is dropped after the split; the fitted medians go to `503-…/imputation_medians.json` and are copied beside `weather_scaler.joblib` (`"imputer"` key). `tests/test_split_imputation.py` pins it.
- Weather values outside `WEATHER_PHYSICAL_BOUNDS` (config) are set to NaN at stage `4b-null_out_of_bounds_weather`, never dropped or clipped, and filled by the post-split imputation. More than 0.1% of a column out of bounds in one month raises (an upstream unit or schema change, not a glitch). The `-1` sentinel of Snow depth and the precipitation columns is inside the limits on purpose. First case: Dew-point -84.8 in the raw 2019_03 file (air 1.5 C, humidity 63% imply -4.8). `tests/test_weather_bounds.py` pins it.
- Cancelled trains stay in the data: stage 12 keeps them although their delay is missing, and `trainDisrupted = delay > TRAIN_DELAY_MINUTES or cancelled` labels them. The threshold lives only in `TRAIN_DELAY_MINUTES` (`DELAY_THRESHOLD_MINUTES` aliases it). A missing delay on a train that was not cancelled is still dropped. `tests/test_cancelled_label.py` pins it.
- Calendar features (`month`, `hour`, `day_of_week`, `day_of_month`) come from `scheduledTime` (`TIME_FEATURE_SOURCE_COLUMN`), never `actualTime`: the actual time is the schedule plus the delay, and one `hour_sin`/`hour_cos` pair scored ROC-AUC 0.775 from it against 0.706 from the schedule. `fcll_sequences._add_temporal_features` mirrors this; `tests/test_time_features.py` pins both.
- Instant weather scenarios classify precipitation from `Precipitation intensity`, never `Precipitation amount` (87.8% null at OL; NaN compares False, so Black Ice fired on 1.4% of rows instead of about 10%). `tests/test_weather_scenarios.py` pins it.
- `differenceInMinutes*` and `year` are never features (they leak the target). The target (`DEFAULT_TARGET_FEATURE`) is kept at stage 502 and every other label column (`LABEL_COLUMNS`) is dropped there, because the trainers use every non-target column as a feature.
- Column names are verbatim (`Air temperature (12h max)`, `wawa_group_<g>`, `weather_scenario[_12h|_24h|_72h]_<Category>`); `fcll_sequences` imports the constants instead of restating them.
- Nothing may block on `input()`: under Slurm stdin is `/dev/null`. `SELECTED_COLUMNS` must be non-empty at run time (`main.py` refuses a training run otherwise) and `SCHEMA_MISMATCH_STRATEGY='intersect'`.
- `fcll_net` is never balanced (`pos_weight` instead of SMOTE), and stage 506 never concatenates all months (64 G OOM).
- One split for every model: the newest year is held out (`*_holdout.parquet`, never trained or tuned on, scored last); the other years are split 80/20 by operating day (`src/day_split.split_days`, shared with `fcll_trainer`). There is no validation split: it used to be written and never read, so 20% of the days were unused. The tabular models validate with `StratifiedGroupKFold` over the train days (`src/cv_utils.py`; stage 503 writes the days to `<base>_train_groups.parquet`, stage 505 copies it beside the scaled train file; a missing sidecar falls back to row-wise CV with a warning, a wrong length raises). `tests/test_grouped_cv.py` pins it.
- `balance_classes` (SMOTE-Tomek) is off and not a `--model` prerequisite: it ran before the CV, so synthetic rows built from validation-fold neighbours were trained on. XGBoost and LightGBM set `scale_pos_weight` to the train neg/pos ratio, Random Forest and Logistic Regression use `class_weight`, Naive Bayes keeps its priors. If it is turned back on the sidecar no longer matches the train rows and CV is row-wise again.
- FCLL leakage masking (`_mask_boundary_crossing_fields`): `run_actual` zeroed on the last Z step, `run_actual` and `dwell_actual` on every H step. Before it the model scored ROC-AUC 0.9993. Scheduled fields stay.
- Only real arrivals at the target are samples; trains that originate there stay in the H pool (otherwise 32% of samples were guaranteed negatives). H is ordered by `visit_time`; `_last_valid` uses the mask cumsum because sequences are pre-padded.
- Binary metrics use `average='binary'` (weighted recall used to equal accuracy). Regression uses wMAPE, not MAPE.
- `log1p` on skewed weather columns must cope with the -1 sentinels (they gave -inf).

## Conventions
- Constants, paths and thresholds live in `config/const*.py`; stage folders are numbered.
- Console output: `"="*60` banners, `→` stage start, `✓`/`✗` result, `⊝ (disabled)` skipped, `⚠️` warning. Stages return dicts with `success`/`error` and append to `result["errors"]`.
- Preprocessing stages raise (or return `None`) on failure and never return the input frame; a missing required input column is a failure. The orchestrator (`execute_preprocessing_pipeline_steps`) alone decides to continue. A failed month removes its stale `training_ready_*`/`preprocessed_data_*` files and `main.py` exits 1.
- Commits: lowercase `area: imperative summary` (`fcll:`, `hpc:`, `fix:`, `docs:`, `test:`, `training:`, `preprocessing:`, `config:`, `env:`, `chore:`). The body explains why, with measured numbers and run/array IDs.
- Bug fixes come with a regression test in `tests/test_<topic>.py` whose module docstring explains the bug.
- Shell-script tests resolve Git Bash through `shutil.which("bash")` because `System32\bash.exe` is WSL; `test_hpc_*` stub `sbatch` on PATH.
- Test patterns: `TrainingPipeline.__new__` with a hand-set `_make_pipeline(tmp_path)`; module reload with `RAILWAY_*` env vars (restore them with the autouse fixture in `test_cli.py`).

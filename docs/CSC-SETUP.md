# Running the Railway-FMI training pipeline on CSC Roihu

Target system is **Roihu**. Puhti and Mahti were retired in July and August 2026.
This workload is **CPU-only** — there is no PyTorch, TensorFlow, or CUDA code in
the repository, so never request `--gres=gpu`.

Replace `<project>` with your CSC project ID (visible in MyCSC) throughout, including
inside the `hpc/*.sh` scripts.

## 1. Connect

```bash
ssh <username>@roihu.csc.fi
```

## 2. Locate your directories

```bash
csc-workspaces
```

You need two:
- `/projappl/<project>` — the Python environment. Persistent.
- `/scratch/<project>` — data and results. **Files untouched for 180 days are deleted.**

Do not run jobs out of `$HOME`; it is quota-limited and not intended for job I/O.

## 3. Clone the repository

```bash
cd /projappl/<project>
git clone <your-repo-url> railway-fmi-code
cd railway-fmi-code
```

`.gitignore` excludes `*.parquet`, so this gets you code only. Data is staged in step 5.

## 4. Build the Python environment with Tykky

Tykky packs the conda environment into a container image, avoiding the
many-small-files penalty conda otherwise incurs on Lustre.

```bash
module load tykky
conda-containerize new --prefix /projappl/<project>/railway-env environment-linux.yml
```

This takes 10-20 minutes. Then:

```bash
export PATH="/projappl/<project>/railway-env/bin:$PATH"
python -c "import sklearn, xgboost, lightgbm, shap, pyarrow, seaborn, haversine; print('env OK')"
```

If a pinned version fails to solve on `linux-64`, relax that single pin in
`environment-linux.yml`, rebuild, and record the deviation at the bottom of this file.

## 5. Stage the data

From your **local machine**, copy the preprocessed training set (~25 MB, 96 files):

```bash
rsync -av "data/output/101-preprocessed_training_ready/" \
    <username>@roihu.csc.fi:/scratch/<project>/railway-fmi/data/output/101-preprocessed_training_ready/
```

Then on Roihu, verify with `hpc/stage_data.sh`, or directly:

```bash
ls -1 /scratch/<project>/railway-fmi/data/output/101-preprocessed_training_ready | wc -l   # expect 96
```

The 2.6 GB `data/input/` is **not** needed: it feeds preprocessing only, which we
run locally.

## 6. Smoke test

```bash
sbatch hpc/smoke_test.sh
squeue --me
```

When it finishes, check the output file:

```bash
cat slurm-smoke-*.out
```

Look for: no `EOFError`, no `UnicodeEncodeError`, no "Enter column numbers" prompt,
and a populated `/scratch/<project>/railway-fmi/data/output/1004-naive_bayes/`.

Then check efficiency:

```bash
seff <jobid>
```

CPU efficiency should be above roughly 70%. Materially lower means thread
oversubscription — revisit the `SEARCH_N_JOBS` / `MODEL_N_JOBS` split in
`config/const_training.py` and the `OMP_NUM_THREADS` exports in the batch script.

## 7. Full run

Five models in parallel, one per array task:

```bash
sbatch hpc/train_array.sh
```

Or all five sequentially in one job:

```bash
sbatch hpc/train_all.sh
```

`train_all.sh` also runs `shap_correlation_analysis`; `train_array.sh` skips it,
because running it once per task would waste allocation.

## 8. Retrieve results

Each model writes a joblib model, JSON metrics, and PNG/PDF figures:

| Directory | Model |
|---|---|
| `data/output/1000-xgboost_randomized_search` | XGBoost |
| `data/output/1001-lightgbm_randomized_search` | LightGBM |
| `data/output/1002-random_forest_randomized_search` | Random Forest |
| `data/output/1003-regularized_regression` | Logistic Regression |
| `data/output/1004-naive_bayes` | Naive Bayes |

Copy them off before the 180-day scratch purge:

```bash
rsync -av <username>@roihu.csc.fi:/scratch/<project>/railway-fmi/data/output/10*/ ./results/
```

## Running a subset

```bash
python main.py --data-root /scratch/<project>/railway-fmi --model xgboost
python main.py --data-root /scratch/<project>/railway-fmi --stages merge_data_files,split_dataset
python main.py --dump-columns          # regenerate the SELECTED_COLUMNS list
```

`--search-iterations` has a minimum of 10; the trainer iterates
`range(10, N+1, 10)`, which is empty below that.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Job hangs, then `EOFError` | `SELECTED_COLUMNS` or `SCHEMA_MISMATCH_STRATEGY` reverted to empty | Restore the frozen values in `config/const_training.py` |
| `UnicodeEncodeError` in the `.out` file | Emoji in `print()` with a non-UTF-8 locale | Confirm `export PYTHONIOENCODING=utf-8` in the batch script |
| `seff` shows CPU efficiency under 30% | Thread oversubscription | Confirm `OMP_NUM_THREADS=1` and that `MODEL_N_JOBS` is 1 |
| "No parquet files found" then exits | Preprocessing enabled with an empty `data/input` | Confirm `EXECUTE_PREPROCESSING_DATA_PIPELINE = False` |
| Trainer runs but produces nothing | `--search-iterations` below 10 | Use 10 or more |
| `sbatch: error: Invalid account` | `<project>` placeholder not replaced | Set your real project ID in all `hpc/*.sh` |

## Environment deviations

Record here any pin relaxed relative to `environment-linux.yml`:

- (none yet)

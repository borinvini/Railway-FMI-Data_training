# Running the Railway-FMI training pipeline on CSC Roihu

Target system is **Roihu**. Puhti and Mahti were retired in July and August 2026.
This workload is **CPU-only** — there is no PyTorch, TensorFlow, or CUDA code in
the repository, so never request `--gres=gpu`.

The CSC project ID `project_2019266` is already filled in throughout this runbook and in the
`hpc/*.sh` scripts. If you switch to a different project, replace it everywhere (including the
`#SBATCH --account=` line and the `PROJECT=` assignment in each script).

The CSC username `vpozzobo` is likewise filled in, both here and as `REMOTE_USER` in
`hpc/stage_data.sh`. Nothing is left to substitute — the scripts are ready to run.

## 0. SSH access (one command, each day you work)

Roihu does **not** accept passwords, and it does not accept a bare public key
either. It requires a short-lived **SSH certificate** signed by CSC. Two facts
that catch people out:

- The login host is `roihu-cpu.csc.fi` — not `roihu.csc.fi`.
- **Certificates expire after 24 hours.** The public key in MyCSC never expires;
  only the certificate does. `Permission denied (publickey)` in the morning is
  routine, not a broken setup.

**Daily:**

```bash
hpc/roihu-auth.sh
```

That signs a fresh certificate, loads your key into the ssh-agent, and prints
the new expiry. It is safe to run repeatedly — when the current certificate is
still valid it exits in about a second without opening a browser. Use
`hpc/roihu-auth.sh -r` to force a re-sign.

Signing opens a MyCSC login in your browser and asks for a 6-digit code. That
step cannot be automated away: CSC publishes no API token for it.

**One-time setup.** If you do not already have the CSC key:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_csc -C "vpozzobo@csc"   # do NOT leave the passphrase empty
```

Register `~/.ssh/id_csc.pub` at [my.csc.fi](https://my.csc.fi) → Profile →
SSH PUBLIC KEYS → **+ Add key**. Then add this stanza to `~/.ssh/config`:

```
Host roihu roihu-cpu.csc.fi
    HostName roihu-cpu.csc.fi
    User vpozzobo
    IdentityFile ~/.ssh/id_csc
    CertificateFile ~/.ssh/id_csc-cert.pub
    IdentitiesOnly yes
    AddressFamily inet
    ServerAliveInterval 60
```

`IdentitiesOnly yes` matters: without it, ssh offers `id_ed25519` (the GitHub
key) first and can exhaust its authentication attempts before reaching `id_csc`.

Finally, so the agent survives between terminals, put in `~/.bashrc`:

```bash
if [ -f "${HOME}/.ssh/agent.env" ]; then
    . "${HOME}/.ssh/agent.env" >/dev/null 2>&1
    # ssh-add exits 2 only when no agent is reachable; 1 just means "no keys yet"
    ssh-add -l >/dev/null 2>&1 || [ "$?" -ne 2 ] || unset SSH_AUTH_SOCK SSH_AGENT_PID
fi
```

Git Bash launches login shells, which do not read `~/.bashrc` on their own —
`~/.bash_profile` must source it. Git for Windows generates a `~/.bash_profile`
that already does.

Without it, the passphrase is retyped in every new terminal. Note this targets
Git Bash's agent, not the Windows `ssh-agent` service — they are separate, and
this project's scripts all run under Git Bash.

**Verify:**

```bash
ssh roihu "echo OK"
```

| Error | Meaning |
|---|---|
| `Permission denied (publickey)` | Certificate expired — run `hpc/roihu-auth.sh` |
| `Network is unreachable` | Usually the wrong hostname, or IPv6 with no IPv6 route — retry with `ssh -4` |
| `Could not resolve hostname` | Typo in the host name |

## 1. Connect

```bash
ssh roihu
```

## 2. Locate your directories

```bash
csc-workspaces
```

You need two:
- `/projappl/project_2019266` — the Python environment. Persistent.
- `/scratch/project_2019266` — data and results. **Files untouched for 180 days are deleted.**

Do not run jobs out of `$HOME`; it is quota-limited and not intended for job I/O.

## 3. Clone the repository

```bash
cd /projappl/project_2019266
git clone <your-repo-url> railway-fmi-code
cd railway-fmi-code
```

`.gitignore` excludes `*.parquet`, so this gets you code only. Data is staged in step 5.

## 4. Build the Python environment with Tykky

Tykky packs the conda environment into a container image, avoiding the
many-small-files penalty conda otherwise incurs on Lustre.

```bash
module load tykky
conda-containerize new --prefix /projappl/project_2019266/railway-env environment-linux.yml
```

This takes 10-20 minutes. Then:

```bash
export PATH="/projappl/project_2019266/railway-env/bin:$PATH"
python -c "import sklearn, xgboost, lightgbm, shap, pyarrow, seaborn, haversine; print('env OK')"
```

If a pinned version fails to solve on `linux-64`, relax that single pin in
`environment-linux.yml`, rebuild, and record the deviation at the bottom of this file.

## 5. Stage the data

From your **local machine**, in the repo root, run the staging script. `PROJECT` and
`REMOTE_USER` are already set in it. It copies the preprocessed training set
(~25 MB, 96 files) to scratch over SSH and prints a file-count and `du -sh` check
at the end:

```bash
hpc/stage_data.sh
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
and a populated `/scratch/project_2019266/railway-fmi/data/output/1000-xgboost_randomized_search/`.

Then check efficiency:

```bash
seff <jobid>
```

The smoke test's own CPU efficiency is not meaningful — 50 fits preceded by a
serial merge/split/SMOTE-Tomek step will read well under 70% by construction.
Use `seff` here only to confirm the job ran and used more than one CPU; judge
efficiency against the 70% rule of thumb on the **full run** (Section 7)
instead, where the 750-fit sweep dominates the walltime. Materially lower than
70% there means thread oversubscription — revisit the `SEARCH_N_JOBS` /
`MODEL_N_JOBS` split in `config/const_training.py` and the `OMP_NUM_THREADS`
exports in the batch script.

## 7. Full run

Five models in parallel, one per array task — the recommended default, since
each task's 8h walltime is checkpointed independently: a timeout or failure in
one model does not cost the other four.

```bash
sbatch hpc/train_array.sh
```

Each array task writes under its own `run_<task-id>/` scratch subdirectory (see
Section 8), to avoid the five tasks racing on shared intermediate files.

Or all five sequentially in one job:

```bash
sbatch hpc/train_all.sh
```

`train_all.sh` also runs `shap_correlation_analysis`; `train_array.sh` skips it,
because running it once per task would waste allocation. `train_all.sh` has no
checkpointing: a timeout or failure anywhere in its single 12h job loses every
model trained so far, not just the one in progress. Prefer `train_array.sh`
unless you specifically need the combined SHAP analysis in one job.

## 8. Retrieve results

`train_array.sh` isolates each array task in its own data root, so results land
under `run_<task-id>/data/output/`, not directly under `data/output/`:

| Directory (relative to `run_<task-id>/data/output/`) | Model | Array task |
|---|---|---|
| `1000-xgboost_randomized_search` | XGBoost | 0 |
| `1001-lightgbm_randomized_search` | LightGBM | 1 |
| `1002-random_forest_randomized_search` | Random Forest | 2 |
| `1003-regularized_regression` | Logistic Regression | 3 |
| `1004-naive_bayes` | Naive Bayes | 4 |

Each writes a joblib model, JSON metrics, and PNG/PDF figures. Copy them all off
before the 180-day scratch purge:

```bash
mkdir -p results
rsync -av vpozzobo@roihu-cpu.csc.fi:/scratch/project_2019266/railway-fmi/run_*/data/output/100[0-4]-*/ ./results/
```

On Windows, Git Bash has no `rsync`. Use `scp` instead:

```bash
mkdir -p results
scp -r "vpozzobo@roihu-cpu.csc.fi:/scratch/project_2019266/railway-fmi/run_*/data/output/100[0-4]-*" ./results/
```

`train_all.sh` does not use per-task run roots (it is a single sequential job,
so there is no race to isolate against), so its results land directly under
`data/output/100[0-4]-*/`:

```bash
mkdir -p results
rsync -av vpozzobo@roihu-cpu.csc.fi:/scratch/project_2019266/railway-fmi/data/output/100[0-4]-*/ ./results/
# or, on Windows Git Bash (no rsync):
scp -r "vpozzobo@roihu-cpu.csc.fi:/scratch/project_2019266/railway-fmi/data/output/100[0-4]-*" ./results/
```

## Running a subset

```bash
python main.py --data-root /scratch/project_2019266/railway-fmi --model xgboost --columns-file config/features.txt
python main.py --dump-columns                        # list every available column
python main.py --columns-file config/features.txt    # select features from a file
```

`--stages` must cover a **contiguous prefix** of the stage chain (merge, filter,
select, split, balance, scale, then a trainer). Disabling `select_training_cols`
while enabling `split_dataset`, for example, makes `split_dataset` fall through
to whatever a previous run already left in its default input folder, splitting
stale or missing data instead of erroring loudly. Prefer `--model`, which always
enables the correct prerequisite chain for you.

`--search-iterations` has a minimum of 10; the trainer iterates
`range(10, N+1, 10)`, which is empty below that.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Job hangs, then `EOFError` | An empty feature selection, or `SCHEMA_MISMATCH_STRATEGY` reverted to empty | Confirm `config/features.txt` is present and non-empty; restore `SCHEMA_MISMATCH_STRATEGY = 'intersect'` |
| `ERROR: columns file missing or empty` | `config/features.txt` was never uploaded to Roihu, or a truncated scp left it zero-byte | Run `hpc/push-features.sh` on your laptop |
| `UnicodeEncodeError` in the `.out` file | Emoji in `print()` with a non-UTF-8 locale | Confirm `export PYTHONIOENCODING=utf-8` in the batch script |
| `seff` shows CPU efficiency under 30% | Thread oversubscription | Confirm `OMP_NUM_THREADS=1` and that `MODEL_N_JOBS` is 1 |
| "No parquet files found" then exits | Preprocessing enabled with an empty `data/input` | Confirm `EXECUTE_PREPROCESSING_DATA_PIPELINE = False` |
| Trainer runs but produces nothing | `--search-iterations` below 10 | Use 10 or more |
| `sbatch: error: Invalid account` | Wrong or expired project, or Roihu access not enabled for it | Check the project in MyCSC and confirm `--account=project_2019266` matches |
| `stage_data.sh` exits "set REMOTE_USER" | `<username>` not replaced | Set `REMOTE_USER` to your CSC username in `hpc/stage_data.sh` |
| `Permission denied (publickey)` | SSH certificate missing or older than 24 h | `hpc/roihu-auth.sh` (Section 0) |
| `ssh: connect ... Network is unreachable` | Wrong host, or IPv6 with no IPv6 route | Use `roihu-cpu.csc.fi`; add `-4` to force IPv4 |

## Environment deviations

Record here any pin relaxed relative to the Windows `environment.yml`:

**2026-08-04 — `psutil` unpinned (was `=5.9.0`).** The first Tykky build failed
with "Could not solve for environment specs": `psutil 5.9.0` has no `cp312` build
on conda-forge/linux-64 (only 3.7–3.10 and pypy), so it could not coexist with
`python=3.12.3`. `psutil` is imported at `src/training_pipeline.py:14` but is not
used for any numerical work, so the version does not affect results.

At the same time, these pure-utility pins were relaxed pre-emptively, to avoid
paying a 10–20 minute rebuild per discovery: `bottleneck`, `numexpr`,
`cloudpickle`, `tqdm`, `python-dateutil`, `pytz`, `packaging`, `six`,
`typing_extensions`. None affects numerical output.

Still pinned exactly, because they *do* affect results: `python=3.12.3`,
`numpy=2.0.1`, `pandas=2.2.3`, `scipy=1.15.1`, `scikit-learn=1.6.1`,
`xgboost=3.0.1`, `shap=0.48.0`, `imbalanced-learn=0.13.0`, `sklearn-compat`,
`slicer`, `numba`, `llvmlite`, `joblib`, `threadpoolctl`.

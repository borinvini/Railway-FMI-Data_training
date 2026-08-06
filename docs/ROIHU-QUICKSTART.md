# Roihu quickstart — resumable

Real values filled in: project `project_2019266`, user `vpozzobo`.
For the reasoning behind any of it, see `CSC-SETUP.md`.

**This file is safe to re-enter at any point.** Every step opens with a CHECK
that tells you whether the step is already done. Run the checks in order, skip
what passes, do the first thing that fails. Nothing here is destructive except
where explicitly marked.

Each block is tagged **[LOCAL]** (your laptop, Git Bash) or **[ROIHU]** (the
login node). Running one in the wrong place is the most common mistake.

---

## Step 0 — [LOCAL] SSH access

**CHECK:**

```bash
ssh roihu "echo OK"
```

`OK` → skip to step 1. `Permission denied (publickey)` → the certificate has
expired. That is routine, not a broken setup: certificates last **24 hours**.

**FIX:**

```bash
hpc/roihu-auth.sh
ssh roihu "echo OK"
```

Signs a fresh certificate, loads the agent, prints the new expiry. Opens a
browser for MyCSC login plus a 6-digit code — unavoidable, CSC offers no API
token. Add `-r` to force a re-sign while the current one is still valid.

Check the expiry any time:

```bash
ssh-keygen -L -f ~/.ssh/id_csc-cert.pub | grep Valid
```

---

## Step 1 — [LOCAL] Local commits pushed

**CHECK:**

```bash
cd "/d/OneDrive - University of Oulu and Oamk/Railway-FMI-Data_training-CSC"
git status --short
echo "local:  $(git rev-parse --short HEAD)"
echo "remote: $(git ls-remote --heads Railway-FMI-Data_training feat/csc-roihu-port | cut -c1-7)"
```

Clean tree and matching hashes → skip to step 2.

**FIX:**

```bash
git add -A && git commit -m "your message"
git push Railway-FMI-Data_training feat/csc-roihu-port
```

---

## Step 2 — [ROIHU] Repository present and current

```bash
ssh roihu
```

**CHECK:**

```bash
cd /projappl/project_2019266/railway-fmi-code 2>/dev/null && \
  git fetch origin -q && \
  echo "branch: $(git branch --show-current)" && \
  echo "local:  $(git rev-parse --short HEAD)" && \
  echo "remote: $(git rev-parse --short origin/feat/csc-roihu-port)"
```

Branch is `feat/csc-roihu-port` and both hashes match → skip to step 3.

⚠️ **If the hashes differ, run `git pull` before doing anything else.** The
`git fetch` above only downloads; your files on disk are still the old version.
Skipping the pull here is how you end up rebuilding the environment from a
stale `environment-linux.yml` and getting an identical failure 20 minutes later.

**FIX — directory missing (never cloned):**

```bash
cd /projappl/project_2019266
git clone https://github.com/borinvini/Railway-FMI-Data_training.git railway-fmi-code
cd railway-fmi-code
git checkout feat/csc-roihu-port
```

**FIX — on `main`, or checkout says `pathspec ... did not match`:** the clone
predates the branch. No need to re-clone:

```bash
git fetch origin
git checkout feat/csc-roihu-port
```

**FIX — hashes differ (behind the remote):**

```bash
git pull
```

---

## Step 3 — [ROIHU] Data staged

**CHECK:**

```bash
ls -1 /scratch/project_2019266/railway-fmi/data/output/101-preprocessed_training_ready 2>/dev/null | wc -l
```

`96` → skip to step 4. Anything else → re-stage from your laptop:

**FIX — [LOCAL]:**

```bash
cd "/d/OneDrive - University of Oulu and Oamk/Railway-FMI-Data_training-CSC"
bash hpc/stage_data.sh
```

---

## Step 4 — [ROIHU] Python environment built

**CHECK:**

```bash
export PATH="/projappl/project_2019266/railway-env/bin:$PATH"
python -c "import sklearn, xgboost, lightgbm, shap, pyarrow, seaborn, haversine, matplotlib; print('env OK')"
```

`env OK` → skip to step 5.

**FIX:** takes 10–20 minutes, so first make sure you are building from the
current `environment-linux.yml`. Step 2's check runs `git fetch`, which only
*downloads* — it does not touch your working tree. If step 2 showed
`local` ≠ `remote`, you still have the old file on disk and the build will fail
in exactly the same way it did before:

```bash
cd /projappl/project_2019266/railway-fmi-code
git pull
git rev-parse --short HEAD    # must now equal the remote hash from step 2
```

⚠️ The `rm -rf` below deletes only the environment directory — a build artifact,
nothing else lives there — and is required because `conda-containerize` refuses
to write into an existing prefix:

```bash
rm -rf /projappl/project_2019266/railway-env
module load tykky
conda-containerize new --prefix /projappl/project_2019266/railway-env environment-linux.yml
export PATH="/projappl/project_2019266/railway-env/bin:$PATH"
python -c "import sklearn, xgboost, lightgbm, shap, pyarrow, seaborn, haversine, matplotlib; print('env OK')"
```

Make the PATH permanent so future logins have it (harmless if repeated — check
first):

```bash
grep -q railway-env ~/.bashrc || echo 'export PATH="/projappl/project_2019266/railway-env/bin:$PATH"' >> ~/.bashrc
```

**If the solve fails** it prints "Could not solve for environment specs" and
names the offending package. Drop that package's `=version` in
`environment-linux.yml`, commit, push, `git pull` on Roihu, and rebuild. Record
it under "Environment deviations" in `CSC-SETUP.md`.

Already hit and fixed: `psutil=5.9.0` has no Python 3.12 build on
conda-forge/linux-64.

---

## Step 5 — [LOCAL] Feature selection uploaded

`config/features.txt` is gitignored, so a fresh clone never has one — the
`git pull`/`git clone` in step 2 does not create it.

**CHECK:**

```bash
ssh roihu "test -s /projappl/project_2019266/railway-fmi-code/config/features.txt && echo OK"
```

`OK` → skip to step 6.

**FIX:**

```bash
cd "/d/OneDrive - University of Oulu and Oamk/Railway-FMI-Data_training-CSC"
cp config/features.example.txt config/features.txt   # first run only
# edit config/features.txt now if you want a different feature set
hpc/push-features.sh
```

This is why no commit is needed when only the feature set changes — see
`ROIHU-RERUN.md`.

---

## Step 6 — [ROIHU] Smoke test

Proves the whole chain cheaply before spending real allocation.

**CHECK:**

```bash
ls /scratch/project_2019266/railway-fmi/data/output/1000-xgboost_randomized_search/ 2>/dev/null
```

Files listed → the smoke test already passed; skip to step 7.

**RUN:**

```bash
cd /projappl/project_2019266/railway-fmi-code
sbatch hpc/smoke_test.sh
squeue --me
```

When it leaves the queue:

```bash
cat slurm-smoke-*.out
```

Success looks like: **no** `EOFError`, **no** `UnicodeEncodeError`, **no**
prompt asking for column numbers, a `Features: N columns, sha256 ...` line
matching what step 5's upload printed, and a populated output directory.

Ignore this job's CPU efficiency — 50 fits behind a serial merge/SMOTE step
reads low by construction. Judge efficiency on the full run.

---

## Step 7 — [ROIHU] Full run

Five models in parallel, one per array task. Preferred: a failure in one model
does not cost the other four.

```bash
cd /projappl/project_2019266/railway-fmi-code
sbatch hpc/train_array.sh
squeue --me
```

Or all five sequentially in one job — also runs the SHAP analysis, but a
timeout loses everything:

```bash
sbatch hpc/train_all.sh
```

---

## Step 8 — [LOCAL] Retrieve results

Git Bash has no `rsync`, so use `scp`:

```bash
cd "/d/OneDrive - University of Oulu and Oamk/Railway-FMI-Data_training-CSC"
mkdir -p results
scp -r "roihu:/scratch/project_2019266/railway-fmi/run_*/data/output/100[0-4]-*" ./results/
```

After `train_all.sh` instead (no per-task run roots):

```bash
scp -r "roihu:/scratch/project_2019266/railway-fmi/data/output/100[0-4]-*" ./results/
```

Scratch deletes anything untouched for 180 days — copy results off.

---

## One-shot state probe [ROIHU]

Paste this whole block to see every step's status at once:

```bash
echo "== repo =="
cd /projappl/project_2019266/railway-fmi-code 2>/dev/null && \
  git fetch origin -q && \
  echo "  branch: $(git branch --show-current)" && \
  echo "  local:  $(git rev-parse --short HEAD)  remote: $(git rev-parse --short origin/feat/csc-roihu-port)" \
  || echo "  NOT CLONED"
echo "== data =="
echo "  files: $(ls -1 /scratch/project_2019266/railway-fmi/data/output/101-preprocessed_training_ready 2>/dev/null | wc -l) (expect 96)"
echo "== env =="
if [ -x /projappl/project_2019266/railway-env/bin/python ]; then
  /projappl/project_2019266/railway-env/bin/python -c "import sklearn,xgboost,lightgbm,shap,pyarrow,seaborn,haversine,matplotlib; print('  env OK')" 2>&1 | tail -1
else
  echo "  NOT BUILT"
fi
echo "== results =="
ls -d /scratch/project_2019266/railway-fmi/*/data/output/100[0-4]-* \
      /scratch/project_2019266/railway-fmi/data/output/100[0-4]-* 2>/dev/null | sed 's/^/  /' || echo "  none yet"
echo "== queue =="
squeue --me
```

---

## Useful while jobs run

```bash
squeue --me                      # your queue
scancel <jobid>                  # cancel one job
seff <jobid>                     # CPU/memory efficiency after it finishes
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
tail -f slurm-train-*.out        # follow a running job
csc-quota                        # disk usage against quota
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Permission denied (publickey)` | Certificate older than 24 h | `hpc/roihu-auth.sh` — step 0 |
| `Network is unreachable` | Wrong host | Use `roihu-cpu.csc.fi` or the `roihu` alias |
| `pathspec ... did not match` | Clone predates the branch | `git fetch origin` then checkout |
| `Could not solve for environment specs` | A pin has no linux-64 build for Python 3.12 | Unpin the named package — step 4 |
| `sbatch: Invalid account` | Project lacks Roihu access | Check MyCSC |
| Job hangs then `EOFError` | Empty feature selection or config reverted to interactive | Check `config/features.txt` is present and non-empty, and `SCHEMA_MISMATCH_STRATEGY` in `config/const_training.py` |
| `ERROR: columns file not found` | `config/features.txt` not uploaded | `hpc/push-features.sh` — see `ROIHU-RERUN.md` step 2 |
| `UnicodeEncodeError` in `.out` | Missing UTF-8 setting | Confirm `PYTHONIOENCODING=utf-8` in the batch script |
| `main.py: No such file` | Wrong directory | `cd /projappl/project_2019266/railway-fmi-code` |
| Trainer runs, produces nothing | `--search-iterations` below 10 | Minimum is 10 |
| Post-quantum SSH warning | Server lacks PQ key exchange | Cosmetic — CSC's to fix, ignore |

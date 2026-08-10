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

`config/features.txt` and `config/scenarios.txt` are both gitignored, so a fresh
clone never has either — the `git pull`/`git clone` in step 2 does not create
them. Upload whichever one the run you intend to submit will read:
`scenarios.txt` for `hpc/train_scenarios.sh`, `features.txt` for
`hpc/train_array.sh`, `hpc/train_all.sh` and `hpc/smoke_test.sh`.

**CHECK — [ROIHU], if you are still on the login node from steps 2-4:**

```bash
test -s /projappl/project_2019266/railway-fmi-code/config/scenarios.txt && echo OK
```

**CHECK — [LOCAL], from your laptop:**

```bash
ssh roihu "test -s /projappl/project_2019266/railway-fmi-code/config/scenarios.txt && echo OK"
```

`OK` → skip to step 6.

⚠️ `roihu` is an alias defined in your laptop's `~/.ssh/config`. Running the
[LOCAL] form while logged into Roihu gives
`ssh: Could not resolve hostname roihu` — use the [ROIHU] form there instead.

**The FIX below must run on your laptop either way.** `config/scenarios.txt` is
gitignored, so it never reaches Roihu through `git pull`; the only way it gets
there is this upload.

**FIX — [LOCAL] — all 8 scenarios (the usual case):**

```bash
cd "/d/OneDrive - University of Oulu and Oamk/Railway-FMI-Data_training-CSC"
cp config/scenarios.example.txt config/scenarios.txt   # first run only
hpc/push-features.sh config/scenarios.txt
```

It prints one line per scenario with its column count. Check the count against
what you expect before submitting 40 jobs.

**FIX — a single ad-hoc feature set (the older flow):**

```bash
cp config/features.example.txt config/features.txt     # first run only
# edit config/features.txt now
hpc/push-features.sh
```

This is why no commit is needed when only the feature set changes — see
`ROIHU-RERUN.md`.

---

## Step 6 — [ROIHU] Pre-flight, two cells

Proves the whole chain cheaply before spending real allocation: same script,
same code path, two tasks instead of forty.

**CHECK:**

```bash
ls -d /scratch/project_2019266/railway-fmi/run_s01_*/data/output/100[0-1]-* 2>/dev/null
```

Two directories listed → the pre-flight already passed; skip to step 7.

**RUN:**

```bash
cd /projappl/project_2019266/railway-fmi-code
sbatch --array=0-1 hpc/train_scenarios.sh
squeue --me
```

Tasks 0 and 1 are scenario 1 with xgboost and lightgbm. When they leave the
queue:

```bash
cat slurm-scenarios-*.out
```

Success looks like: **no** `EOFError`, **no** `UnicodeEncodeError`, **no**
prompt asking for column numbers, a `Features: config/scenarios.txt scenario
1/8, sha256 ...` line whose hash matches what step 5's upload printed, and two
populated `run_s01_*` roots.

If the sha256 does **not** match, the cluster is training on a stale catalogue —
re-run step 5 before going any further, or all 40 results will be labelled with
feature sets they were not trained on.

Ignore this job's CPU efficiency — a short search behind a serial merge/SMOTE
step reads low by construction. Judge efficiency on the full run.

**Running the older single-feature-set flow instead?** Its equivalent is
`sbatch hpc/smoke_test.sh`, which reads `config/features.txt` (not the
catalogue) and trains one xgboost model. It fails with `columns file missing or
empty` if you have only uploaded `config/scenarios.txt`.

---

## Step 7 — [ROIHU] Full run

**Check disk headroom first.** 40 tasks each keep their own copy of the prep
stages — eight times the footprint of the old five-task array.

Roihu's default scratch quota is **250 GiB**
([CSC docs](https://docs.csc.fi/accounts/how-to-increase-disk-quotas/)). There
is no `csc-quota` command on Roihu; that tool is Puhti/Mahti only, and quotas
are viewed and changed in MyCSC (project → Configuration). Measure instead:

```bash
du -sh /scratch/project_2019266/railway-fmi/run_0        # one root from the old flow
du -sh /scratch/project_2019266/railway-fmi              # everything already there
```

Multiply the first by 40 and check it fits in 250 GiB alongside the second.
Under ~5 GB per root is comfortable; above ~6 GB you will exceed the quota
part-way through the array and lose the tail of the run.

If it does not fit, in order of preference: delete the old `run_0`..`run_4`
roots once their results are fetched; or raise the quota in MyCSC — but note
that an increased quota bills Storage Billing Units on the **quota**, not on
what you actually store, and scratch's automatic cleaning keeps removing idle
files regardless.

All 8 scenarios × 5 models, 40 independent tasks:

```bash
cd /projappl/project_2019266/railway-fmi-code
sbatch hpc/train_scenarios.sh
squeue --me
```

Step 6 is the two-cell pre-flight; re-run it whenever anything upstream changes.

Any single cell can be re-run on its own — the roots are independent:

```bash
sbatch --array=16 hpc/train_scenarios.sh     # scenario 4, lightgbm
```

Task id decomposes as `scenario = id / 5 + 1`, `model = id % 5` over
`(xgboost lightgbm random_forest logistic_regression naive_bayes)`.

The older single-feature-set flows still work unchanged:

```bash
sbatch hpc/train_array.sh    # 5 models, config/features.txt
sbatch hpc/train_all.sh      # 5 models sequentially, plus the SHAP analysis
```

---

## Step 8 — [LOCAL] Retrieve results

After `hpc/train_scenarios.sh`:

```bash
cd "/d/OneDrive - University of Oulu and Oamk/Railway-FMI-Data_training-CSC"
hpc/fetch-results.sh --scenarios
```

That pulls the 40 model directories and sorts them into one folder per
scenario, named as the scenario is named in `config/scenarios.txt`:

```
results/
  1 - ALL FEATURES (OPERACIONAL + INSTANT WEATHER + ALL ROLLING WINDOWS + WEATHER SCENARIOS)/
      1000-xgboost_randomized_search/
      1001-lightgbm_randomized_search/
      1002-random_forest_randomized_search/
      1003-regularized_regression/
      1004-naive_bayes/
  2 - ONLY OPERACIONAL FEATURES/
      ...
```

Add `--stages` for the intermediate datasets too — one set per scenario, which
is a much larger transfer:

```bash
hpc/fetch-results.sh --scenarios --stages
```

After `hpc/train_array.sh` (the 5-job flow), unchanged:

```bash
hpc/fetch-results.sh              # every stage
hpc/fetch-results.sh --models     # model directories only
hpc/fetch-results.sh --train-all  # after train_all.sh
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
ls -d /scratch/project_2019266/railway-fmi/run_s??_*/data/output/100[0-4]-* \
      /scratch/project_2019266/railway-fmi/run_*/data/output/100[0-4]-* \
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
tail -f slurm-scenarios-*.out    # follow a running scenario task
tail -f slurm-train-*.out        # follow a train_array.sh / train_all.sh job
du -sh /scratch/project_2019266/railway-fmi   # scratch usage (quota is 250 GiB;
                                              # no csc-quota on Roihu — see MyCSC)
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
| `ERROR: columns file missing or empty` | `config/features.txt` not uploaded, or a truncated scp left it zero-byte | `hpc/push-features.sh` — see `ROIHU-RERUN.md` step 2 |
| `UnicodeEncodeError` in `.out` | Missing UTF-8 setting | Confirm `PYTHONIOENCODING=utf-8` in the batch script |
| `main.py: No such file` | Wrong directory | `cd /projappl/project_2019266/railway-fmi-code` |
| Trainer runs, produces nothing | `--search-iterations` below 10 | Minimum is 10 |
| Post-quantum SSH warning | Server lacks PQ key exchange | Cosmetic — CSC's to fix, ignore |
| `contains 8 scenario sections — pass --scenario` | `--columns-file` pointed at the catalogue with no scenario chosen | Add `--scenario <index>`, or use `config/features.txt` |
| `scenario index N is out of range` | Array range and catalogue size disagree | The error names the correct `--array` range; resubmit with it |
| `results/` holds `run_sNN_*` folders | No python on PATH during the fetch | Re-run `hpc/fetch-results.sh --scenarios` with python available |

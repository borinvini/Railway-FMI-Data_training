# Roihu re-run — the everyday loop

Setup is already done. This is the short cycle you repeat each time you want new
results: certificate → features/push → pull → clean → run → fetch.

For first-time setup see `ROIHU-QUICKSTART.md`. For the reasoning behind any
step, `CSC-SETUP.md`.

Blocks are tagged **[LOCAL]** (your laptop, Git Bash) or **[ROIHU]** (login
node). Values are filled in: project `project_2019266`, user `vpozzobo`.

---

## The whole loop, if nothing is broken

```bash
# [LOCAL] scenarios or features only — no commit needed
cd "/d/OneDrive - University of Oulu and Oamk/Railway-FMI-Data_training-CSC"
# first run only: cp config/scenarios.example.txt config/scenarios.txt
hpc/push-features.sh config/scenarios.txt

# [LOCAL] code changes — these still go through git
git add -A && git commit -m "your message"
git push Railway-FMI-Data_training feat/csc-roihu-port
ssh roihu
```

```bash
# [ROIHU] pull, clean, run
cd /projappl/project_2019266/railway-fmi-code
git pull
rm -rf /scratch/project_2019266/railway-fmi/run_*
rm -f slurm-train-*.out
sbatch hpc/train_scenarios.sh    # all 8 scenarios x 5 models
sbatch hpc/train_array.sh        # or: single feature set, 5 models
squeue --me
```

```bash
# [LOCAL] once the queue is empty
cd "/d/OneDrive - University of Oulu and Oamk/Railway-FMI-Data_training-CSC"
hpc/fetch-results.sh
```

The rest of this file explains each step and what to do when one fails.

---

## 1. [LOCAL] Certificate

The SSH certificate lasts **24 hours**. Expiry looks like a broken setup but is
routine.

```bash
hpc/roihu-auth.sh
```

Signs only if needed — when the current certificate is still valid it exits in
about a second without opening a browser, so running it every morning costs
nothing. It also loads `id_csc` into the ssh-agent, so the passphrase is entered
once per boot rather than once per command. Add `-r` to force a re-sign.

Confirm:

```bash
ssh roihu "echo OK"
```

---

## 2. [LOCAL] Push your changes

### Features — no commit

`config/features.txt` is gitignored. Edit it, then upload it straight into the
cluster's clone:

```bash
cd "/d/OneDrive - University of Oulu and Oamk/Railway-FMI-Data_training-CSC"
hpc/push-features.sh
```

One name per line, taken verbatim — no quotes, no commas. `trainDelayed` must
stay: it is the target, and the trainers split it out by name.
`python main.py --dump-columns` lists everything available, and
`config/features.example.txt` is the committed starting point if you need one.

The script prints a sha256. The batch scripts print the same one into the slurm
log, so you can confirm afterwards which selection actually ran.

No `config/features.txt` yet? `cp config/features.example.txt config/features.txt`.

`config/scenarios.txt` is gitignored for the same reason and uploads the same
way, with the path given explicitly: `hpc/push-features.sh config/scenarios.txt`.

### Code — commit as before

The cluster reads its own clone, so anything you change under `src/`, `config/`
(other than the feature file) or `hpc/` has to reach GitHub first.

```bash
git status --short
git add -A && git commit -m "your message"
git push Railway-FMI-Data_training feat/csc-roihu-port
```

Changed only the features? The push and the pull in step 3 are both unnecessary
— skip to step 4.

---

## 3. [ROIHU] Pull — the step that is never optional

```bash
ssh roihu
cd /projappl/project_2019266/railway-fmi-code
git pull
git log --oneline -1
```

That hash must match what you just pushed. Skipping this is the single most
expensive mistake available here: the job runs happily against the *old* code
and you burn an hour of allocation to reproduce results you already had.

### [LOCAL] Confirm the two sides actually match

⚠️ **This one runs on your laptop, not on Roihu** — it is the only `[LOCAL]`
block in this section. `roihu` is a `Host` alias in your laptop's
`~/.ssh/config`; Roihu has no such alias, so running this there fails with
`ssh: Could not resolve hostname roihu`.

From the repo root on your laptop, after the pull:

```bash
LOCAL_HEAD="$(git rev-parse HEAD)"
ROIHU_HEAD="$(ssh roihu 'cd /projappl/project_2019266/railway-fmi-code && git rev-parse HEAD')"

if [ -z "${ROIHU_HEAD}" ]; then
    echo "CHECK FAILED — could not read Roihu's HEAD. Nothing was compared."
    echo "  Are you running this on your laptop? Is the certificate current?"
elif [ "${LOCAL_HEAD}" = "${ROIHU_HEAD}" ]; then
    echo "IN SYNC   ${LOCAL_HEAD}"
else
    echo "OUT OF SYNC — git pull on Roihu"
    echo "  laptop: ${LOCAL_HEAD}"
    echo "  roihu : ${ROIHU_HEAD}"
fi
```

The three-way branch matters. A one-liner that just compares the two values
reports `OUT OF SYNC` whenever `ssh` fails, because the comparison is then
against an empty string — telling you the clones disagree when it has in fact
learned nothing. `CHECK FAILED` and `OUT OF SYNC` need different fixes.

Roihu's sshd prints three `** WARNING: ... post-quantum ...` lines on every
connection. They go to stderr, so they do not affect the comparison — ignore
them.

**Already logged into Roihu?** Then just run `git rev-parse HEAD` there and
compare it against `git rev-parse HEAD` on your laptop by eye. There is no way
to reach your laptop from Roihu.

Two things this does **not** prove, both worth a second command when results
look wrong:

- **Matching hashes do not mean matching files.** A commit match covers tracked
  files only. Someone can edit `config/const_training.py` directly on Roihu and
  the hashes still agree. Check with `git status --porcelain` there — anything
  beyond untracked `slurm-*.out` job logs means the clone has drifted.
- **A pull only helps if the push landed first.** If you skipped step 2, Roihu
  pulls nothing and reports success. The check above catches this, because your
  local `HEAD` will be ahead of Roihu's.

The symptom of getting this wrong is silent: results identical to your last run,
with no error anywhere.

---

## 4. [ROIHU] Clean the previous run

Not strictly required — outputs are overwritten in place — but recommended,
because per-stage logs open in append mode and old artifacts are
indistinguishable from new ones if a task fails halfway.

```bash
rm -rf /scratch/project_2019266/railway-fmi/run_*
rm -f slurm-train-*.out slurm-smoke-*.out
```

⚠️ **Never delete `/scratch/project_2019266/railway-fmi/data/`.** That is the 96
staged input files; losing them means re-staging from your laptop. Everything
the pipeline generates lives under `run_*/`, so the command above is enough.

`rm -rf run_*` is safe even though each run directory contains
`101-preprocessed_training_ready`: that is a symlink, and `rm` unlinks it rather
than following it to the real data. Confirm if you want to be sure:

```bash
ls -1 /scratch/project_2019266/railway-fmi/data/output/101-preprocessed_training_ready | wc -l   # 96
```

---

## 5. [ROIHU] Optional smoke test

Worth the ~5 minutes whenever you changed `config/features.txt` or anything in
`src/`. It catches a bad column name immediately instead of across five array
tasks.

```bash
sbatch hpc/smoke_test.sh
squeue --me
cat slurm-smoke-*.out
```

---

## 6. [ROIHU] Run

Five models in parallel, one per array task:

```bash
cd /projappl/project_2019266/railway-fmi-code
sbatch hpc/train_array.sh
squeue --me
```

| Array task | Model |
|---|---|
| `_0` | XGBoost |
| `_1` | LightGBM |
| `_2` | Random Forest |
| `_3` | Logistic Regression |
| `_4` | Naive Bayes |

Expect roughly 10–20 minutes for tasks 0, 1 and 4, and **1–2 hours** for tasks 2
and 3 — Random Forest fits hundreds of trees per candidate, and Logistic
Regression uses the slow `saga` solver needed for the l1/elasticnet search.

Follow one live:

```bash
tail -f slurm-train-*_2.out
```

---

## 7. [ROIHU] Confirm it actually worked

Leaving the queue is not the same as succeeding.

**While it runs — how many tasks are actually left.** Plain `squeue --me`
collapses pending array tasks into a single row, so `563786_[33-39]` looks like
one job when it is seven. `-r` expands them:

```bash
squeue --me -r -h | wc -l        # tasks still queued or running
squeue --me -r -h -t R | wc -l   # running right now
squeue --me -r -h -t PD | wc -l  # still pending
```

Finished tasks leave the queue, so these counts fall below 40 as the array
progresses, and gaps appear in the running ids (0-8, then 10-...). A gap means
that cell is done, not lost.

**Once it drains — the tally that matters:**

```bash
sacct -j <arrayjobid> -X --format=State -n | sort | uniq -c
```

One line per outcome. `40 COMPLETED` is the goal; `-X` gives one row per task
rather than one per job step. Anything else, drill into the specific cell:

```bash
sacct -j <arrayjobid>_<taskid> --format=JobID,State,Elapsed,ExitCode,MaxRSS
tail -30 slurm-scenarios-<arrayjobid>_<taskid>.out
```

Want `COMPLETED` and `0:0`. `FAILED`, or any non-zero ExitCode, means at least
one model did not train. `OUT_OF_MEMORY` or a `MaxRSS` near the 32G request
means raise `--mem`, not re-run as-is.

Map a task id back to what it was training: `scenario = id / 5 + 1`,
`model = id % 5` over
`(xgboost lightgbm random_forest logistic_regression naive_bayes)`. Task 9 is
scenario 2, naive_bayes.

**Then confirm the output really exists** — a task can exit 0 and still leave an
empty directory:

```bash
ls /scratch/project_2019266/railway-fmi/run_s??_*/data/output/100[0-4]-*/   # scenario runs
ls /scratch/project_2019266/railway-fmi/run_*/data/output/100[0-4]-*/       # train_array.sh runs
```

Each directory should hold a `.joblib` model, a metrics `.json`, and PNG/PDF
figures. An empty directory means that model failed regardless of what the
terminal implied.

Re-run only the cells that failed — the roots are independent:

```bash
sbatch --array=9,17,23 hpc/train_scenarios.sh
```

---

## 8. [LOCAL] Fetch the results

```bash
cd "/d/OneDrive - University of Oulu and Oamk/Railway-FMI-Data_training-CSC"
hpc/fetch-results.sh
```

This brings back **every** training-pipeline stage, not only the models:

| Directory | What it holds |
|---|---|
| `500-merge_data_files` | The merged dataset, all months in one frame |
| `501-filter_delay_outliers` | After the quantile cut on the delay tails |
| `502-select_training_cols` | After `config/features.txt` was applied — differs per scenario under `train_scenarios.sh` |
| `503-split_dataset` | train / test / out-of-time holdout |
| `504-balance_classes` | After SMOTE-Tomek |
| `505-scale_weather_features` | The frame the trainers actually saw |
| `700-shap_correlation_analysis` | Only after `train_all.sh` |
| `1000-*` … `1004-*` | One per model: joblib, metrics JSON, figures |

Stages 500-505 come from `run_0` alone — that applies to `train_array.sh`
only. `train_array.sh` re-runs them in all five run roots from identical code
and identical input, so the other four copies are the same bytes and five
times the download. The model directories come from all five roots, since each
task produces exactly one.

After `hpc/train_scenarios.sh`, the equivalent single-root shortcut is
`run_s??_xgboost` per scenario: `502-select_training_cols` onward genuinely
differs per scenario, which is why `--stages` there returns eight sets rather
than one.

The script prints what it is about to copy and how big it is before it starts —
this is a much larger transfer than the models alone. Variants:

```bash
hpc/fetch-results.sh --scenarios          # 40 model directories, sorted per scenario
hpc/fetch-results.sh --scenarios --stages # + one prep-stage set per scenario
hpc/fetch-results.sh --models             # only 1000-1004, the old behaviour
hpc/fetch-results.sh --train-all          # after train_all.sh, which has no run_N roots
```

This **overwrites** whatever is already in `results/`. Keep a previous run by
renaming first:

```bash
mv results results-$(date +%Y%m%d)
```

Scratch deletes anything untouched for 180 days, so copy results off.

---

## Compare runs

Metrics live in `results/*/[model]_iteration_analysis_selected.json`. This prints
the headline table:

```bash
python -c "
import json, glob, os
print(f'{\"model\":<22}{\"TEST f1\":>9}{\"TEST auc\":>9}{\"HOLD f1\":>9}{\"HOLD auc\":>9}')
for f in sorted(glob.glob('results/*/*_iteration_analysis_selected.json')):
    d = json.load(open(f))
    n = os.path.basename(os.path.dirname(f)).split('-', 1)[1][:21]
    t, h = d['final_metrics'], d['holdout_metrics']
    print(f\"{n:<22}{t['test_f1']:9.4f}{t['test_auc']:9.4f}{h['f1']:9.4f}{h['auc']:9.4f}\")
"
```

Judge on **HOLDOUT**, not TEST. The holdout is the out-of-time final year and is
the honest estimate of how the model behaves on data from a period it never saw.

---

## Changing what gets trained

**Which features** — `config/features.txt`, one name per line, then
`hpc/push-features.sh`. No commit, no pull. `'trainDelayed'` must stay: it is
the target, and the trainers split it out by name. `python main.py
--dump-columns` lists everything available. With no `--columns-file`, the
frozen fallback in `config/const_training.py:69` applies instead.

**Which scenario** — `config/scenarios.txt`, then
`hpc/push-features.sh config/scenarios.txt`. No commit, no pull. List them
with `python main.py --columns-file config/scenarios.txt --list-scenarios`.
Run one locally with `--scenario <index>`.

**Which models** — edit `MODELS=(...)` in `hpc/train_array.sh:65` and match
`--array=0-N` to its length. Or run one directly:

```bash
python main.py --data-root /scratch/project_2019266/railway-fmi/run_0 --model xgboost --columns-file config/features.txt
```

**Search budget** — `RANDOM_SEARCH_ITERATIONS` in `config/const_training.py`.
Minimum 10; the trainer iterates `range(10, N+1, 10)`, which is empty below that.

**Cores per task** — `--cpus-per-task` in `hpc/train_array.sh`. `SEARCH_N_JOBS`
picks it up automatically. Check `seff <jobid>` first: if efficiency is already
below ~50% you are bounded by the serial merge/SMOTE stages and more cores will
not help.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Permission denied (publickey)` | Certificate older than 24 h | `hpc/roihu-auth.sh` — step 1 |
| Results identical to last run | Forgot `git pull` on Roihu | Step 3 |
| `pathspec ... did not match` | Clone predates a new branch | `git fetch origin` then checkout |
| `sacct` shows FAILED | A model crashed | Read that task's `slurm-train-*_N.out` |
| Empty model directory | That task failed | Same as above |
| `main.py: No such file` | Submitted from the wrong directory | `cd` to the repo root before `sbatch` |
| Job hangs then `EOFError` | Config reverted to interactive | Check `config/features.txt` reached Roihu, and `SCHEMA_MISMATCH_STRATEGY` |
| `columns file missing or empty` | `features.txt` never uploaded, or a truncated scp left it zero-byte | `hpc/push-features.sh` from your laptop |
| `SELECTED_COLUMNS references columns not in DataFrame` | Typo or a dropped column | Re-check with `--dump-columns` |
| Features not what you expected | An older `features.txt` on Roihu | Compare the log's sha256 with `sha256sum config/features.txt` |
| Trainer runs but produces nothing | `--search-iterations` below 10 | Use 10 or more |
| Post-quantum SSH warning | Server lacks PQ key exchange | Cosmetic, ignore |

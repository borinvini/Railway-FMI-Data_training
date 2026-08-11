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
# [LOCAL] scenarios or features — only if you edited them since the last push
cd "/d/OneDrive - University of Oulu and Oamk/Railway-FMI-Data_training-CSC"
# first run only: cp config/scenarios.example.txt config/scenarios.txt
hpc/push-features.sh config/scenarios.txt

# [LOCAL] code changes — only if you have something uncommitted or unpushed
git status -sb    # no file lines and no "ahead"? skip the next two commands
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
# [LOCAL] once the queue is empty — the flag must match the script you ran
cd "/d/OneDrive - University of Oulu and Oamk/Railway-FMI-Data_training-CSC"
scp roihu:/projappl/project_2019266/railway-fmi-code/config/scenarios.txt config/
hpc/fetch-results.sh --scenarios   # after train_scenarios.sh
hpc/fetch-results.sh               # after train_array.sh
```

**Both `[LOCAL]` push blocks are conditional. The `git pull` is not.** Skip
`push-features.sh` when you have not touched the feature or scenario file, and
skip the commit/push when `git status -sb` shows a clean tree with no `ahead`
marker — pushing again with nothing to send is a no-op, not a safety net.

But having pushed — a minute ago or last week — is not a reason to skip
`git pull` on Roihu. The push moves your commits to GitHub; only the pull moves
them into the cluster's clone. Those are two different machines, and the job
reads the second one. Step 3 is the one command in this loop that runs every
single time.

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

### Already committed and pushed earlier? Skip this whole step

Common when you did the work in one sitting and only now got round to running
the job. One command tells you:

```bash
git status -sb | head -1
```

```
## feat/csc-roihu-port...Railway-FMI-Data_training/feat/csc-roihu-port
```

No `[ahead N]` on that line and no files listed underneath means everything is
already on GitHub — there is nothing to commit and nothing to push. Go to
step 3. `[ahead 2]` means two commits never left your laptop; push them.

Plain `git push` works too if the branch has an upstream, which this one does —
it resolves to the same remote and branch the long form spells out.

Changed only `features.txt` or `scenarios.txt`? Those are gitignored, so there
is nothing for git to carry either way — `hpc/push-features.sh` in the previous
section already delivered them.

**None of this excuses step 3.** Whether you pushed a minute ago, pushed last
week, or had nothing to push at all, Roihu's clone only advances when you run
`git pull` on Roihu. A pull with nothing to fetch costs a second and prints
`Already up to date.`; a pull you skipped costs an hour of allocation spent
re-running the old code. Run it every time.

---

## 3. [ROIHU] Pull — the step that is never optional

```bash
ssh roihu
cd /projappl/project_2019266/railway-fmi-code
git pull
git log --oneline -1
```

That hash must match your laptop's `git rev-parse HEAD` — not "what you just
pushed", because you may have pushed days ago and the point is the same either
way. Run this even when step 2 had nothing to do. Skipping it is the single most
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
ls /scratch/project_2019266/railway-fmi/run_s??_*/data/output/100[0-4]-*/    # scenario runs
ls /scratch/project_2019266/railway-fmi/run_[0-9]/data/output/100[0-4]-*/    # train_array.sh runs
```

Each directory should hold **six** files: a `_best_model_selected.pkl`, an
`_iteration_analysis_selected.json` of metrics, a `_feature_importance_selected.csv`,
and three PNGs. An empty or short directory means that model failed regardless
of what the terminal implied.

This counts the whole 40-task array in one line — `40` means every cell
produced a model:

```bash
ls -d /scratch/project_2019266/railway-fmi/run_s??_*/data/output/100[0-4]-*/*_best_model_selected.pkl | wc -l
```

Note the `run_[0-9]` above rather than `run_*`: the bare `run_*` also matches
the `run_sNN_<model>` scenario roots, so it silently mixes two different runs
into one listing.

Re-run only the cells that failed — the roots are independent:

```bash
sbatch --array=9,17,23 hpc/train_scenarios.sh
```

---

## 8. [LOCAL] Fetch the results

After `hpc/train_scenarios.sh` — the 40-task array — this is the whole step:

```bash
cd "/d/OneDrive - University of Oulu and Oamk/Railway-FMI-Data_training-CSC"
scp roihu:/projappl/project_2019266/railway-fmi-code/config/scenarios.txt config/
hpc/fetch-results.sh --scenarios
```

**Do not run `hpc/fetch-results.sh` with no flags after a scenario array.** Its
default glob is `run_*`, which also matches `run_s01_xgboost`, and the default
mode archives by basename alone. All eight scenarios' `1000-xgboost_randomized_search`
then land on one path in `results/`, each overwriting the last: 40 directories
arrive, 5 survive, and nothing says which scenario won. `--scenarios` keeps the
`run_sNN_` prefix through the transfer and sorts it out locally.

The `scp` line is not optional, and it is not fetching results — it is fetching
the **catalogue**. `config/scenarios.txt` is gitignored and uploaded straight to
the cluster, so a fresh clone does not have it. Without it the transfer still
succeeds but the sort is skipped, leaving `results/run_s01_xgboost/...` slugs
instead of scenario names — and the "Compare runs" snippet below globs
`results/*/100[0-4]-*/`, which a slugged root's extra `data/output/` levels do
not match, so it finds nothing.

The sort is only as honest as the catalogue it uses. Confirm the copy you just
pulled is the one the cluster actually trained against — these two must print
the same 16 characters:

```bash
sha256sum config/scenarios.txt | cut -c1-16
ssh roihu 'grep -h "^Features:" /projappl/project_2019266/railway-fmi-code/slurm-scenarios-<arrayjobid>_*.out | sed "s/.*sha256 //" | sort -u'
```

One line out of the second command means all 40 tasks agreed. If they differ,
the catalogue was edited mid-run and the folder names do not describe what was
trained.

The prep stages make this a bigger transfer than the models alone —
`500-merge_data_files` holds the full merged dataset, once per scenario. The
script prints the total size before it moves a single byte, so read that line
and Ctrl-C if it is more than you meant to pull.
`hpc/fetch-results.sh --scenarios --models` fetches the models alone.

**What you should end up with:**

```bash
ls -d results/*/ | wc -l                       # 8   — one folder per scenario
ls -d results/*/100[0-4]-*/ | wc -l            # 40  — five model dirs per scenario
ls -d results/*/50[0-5]-*/ | wc -l             # 48  — one prep set per scenario
find results -type d -empty                    # nothing
find results -name '*_best_model_selected.pkl' | wc -l   # 40
```

Folders are named from the catalogue, e.g. `results/2 - ONLY OPERACIONAL FEATURES/`.
Each holds one flat set of eleven directories — the six stages that scenario was
prepared from, and the five models trained out of them:

```
results/2 - ONLY OPERACIONAL FEATURES/
├── 500-merge_data_files/
├── 501-filter_delay_outliers/
├── 502-select_training_cols/
├── 503-split_dataset/
├── 504-balance_classes/
├── 505-scale_weather_features/
├── 1000-xgboost_randomized_search/       ← the six files below
├── 1001-lightgbm_randomized_search/
├── 1002-random_forest_randomized_search/
├── 1003-regularized_regression/
└── 1004-naive_bayes/
```

**One prep set per scenario, not per model.** All five models in a scenario
re-run stages 500-505 from identical code over identical input, so the fetch
takes them from that scenario's `run_s??_xgboost` root alone — the other four
copies would be the same bytes and five times the transfer. That is also what
makes the flat layout safe: only one root per scenario carries a
`500-merge_data_files`, so nothing collides on arrival.

The saving does not extend across scenarios. `502-select_training_cols` onward
sits downstream of the feature selection, so each scenario's split, balance and
scaling genuinely differ; only `500-` and `501-` are identical everywhere. That
is why it is eight sets rather than one.

Each `100[0-4]-*` directory holds exactly six files:

| File | What it is |
|---|---|
| `<model>_best_model_selected.pkl` | The fitted model |
| `<model>_iteration_analysis_selected.json` | Metrics — what "Compare runs" reads |
| `<model>_iteration_analysis_selected.png` | Search iterations plotted |
| `<model>_feature_importance_selected.csv` | Importances, ranked |
| `<model>_feature_importance_selected.png` | The same, plotted |
| `<model>_confusion_matrix_selected.png` | Test-set confusion matrix |

Fewer than six files, or an empty directory, means that cell failed no matter
what Slurm reported — go back to step 7 and re-run it.

### Older layouts

`hpc/train_array.sh` (5 tasks, one model each, `run_0`..`run_4`) is what the
bare command was written for:

```bash
hpc/fetch-results.sh
```

That brings back **every** training-pipeline stage, not only the models:

| Directory | What it holds |
|---|---|
| `500-merge_data_files` | The merged dataset, all months in one frame |
| `501-filter_delay_outliers` | After the quantile cut on the delay tails |
| `502-select_training_cols` | After `config/features.txt` was applied — differs per scenario under `train_scenarios.sh` |
| `503-split_dataset` | train / test / out-of-time holdout |
| `504-balance_classes` | After SMOTE-Tomek |
| `505-scale_weather_features` | The frame the trainers actually saw |
| `700-shap_correlation_analysis` | Only after `train_all.sh` |
| `1000-*` … `1004-*` | One per model: the six files listed above |

In this older layout stages 500-505 come from `run_0` alone. `train_array.sh`
re-runs them in all five run roots from identical code and identical input, so
the other four copies are the same bytes and five times the download. There is
only ever one scenario in play, so one set is all there is.

`--scenarios` applies the same rule one level down: one set per scenario rather
than one overall, taken from each scenario's `run_s??_xgboost` root.

The script prints what it is about to copy and how big it is before it starts —
the stages are a much larger transfer than the models alone. All variants:

```bash
hpc/fetch-results.sh --scenarios          # 40 model dirs + 8 prep sets, sorted per scenario
hpc/fetch-results.sh --scenarios --models # the 40 model directories alone
hpc/fetch-results.sh                      # train_array.sh: 5 models + run_0 stages
hpc/fetch-results.sh --models             # train_array.sh: only 1000-1004
hpc/fetch-results.sh --train-all          # after train_all.sh, which has no run_N roots
```

`--models` subtracts from the default in every mode. There used to be a
`--stages` flag that added the prep stages to `--scenarios`; the stages now come
down by default, so the flag is accepted, ignored, and can be dropped from any
command you have written down.

This **overwrites** whatever is already in `results/`. Keep a previous run by
renaming first:

```bash
mv results results-$(date +%Y%m%d)
```

Scratch deletes anything untouched for 180 days, so copy results off.

---

## Compare runs

Metrics live in
`results/<scenario>/100[0-4]-*/[model]_iteration_analysis_selected.json`. The
`100[0-4]-` in the glob is what keeps the prep stages out of the table — they
sit flat beside the model directories and hold no metrics, only the frames the
trainers were handed. This prints the headline table, all 40 rows:

```bash
python -c "
import json, glob, os
print(f'{\"scenario\":<28}{\"model\":<22}{\"TEST f1\":>9}{\"TEST auc\":>9}{\"HOLD f1\":>9}{\"HOLD auc\":>9}')
for f in sorted(glob.glob('results/*/100[0-4]-*/*_iteration_analysis_selected.json')):
    d = json.load(open(f))
    m = os.path.basename(os.path.dirname(f)).split('-', 1)[1][:21]
    s = os.path.basename(os.path.dirname(os.path.dirname(f)))[:27]
    t, h = d['final_metrics'], d['holdout_metrics']
    print(f\"{s:<28}{m:<22}{t['test_f1']:9.4f}{t['test_auc']:9.4f}{h['f1']:9.4f}{h['auc']:9.4f}\")
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

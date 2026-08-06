# Roihu re-run — the everyday loop

Setup is already done. This is the short cycle you repeat each time you want new
results: certificate → push → pull → clean → run → fetch.

For first-time setup see `ROIHU-QUICKSTART.md`. For the reasoning behind any
step, `CSC-SETUP.md`.

Blocks are tagged **[LOCAL]** (your laptop, Git Bash) or **[ROIHU]** (login
node). Values are filled in: project `project_2019266`, user `vpozzobo`.

---

## The whole loop, if nothing is broken

```bash
# [LOCAL] push whatever you changed
cd "/d/OneDrive - University of Oulu and Oamk/Railway-FMI-Data_training-CSC"
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
sbatch hpc/train_array.sh
squeue --me
```

```bash
# [LOCAL] once the queue is empty
cd "/d/OneDrive - University of Oulu and Oamk/Railway-FMI-Data_training-CSC"
mkdir -p results
scp -r "roihu:/scratch/project_2019266/railway-fmi/run_*/data/output/100[0-4]-*" ./results/
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

The cluster reads its own clone. Anything you edit locally — most often
`SELECTED_COLUMNS` in `config/const_training.py` — has to reach GitHub first.

```bash
cd "/d/OneDrive - University of Oulu and Oamk/Railway-FMI-Data_training-CSC"
git status --short
git add -A && git commit -m "your message"
git push Railway-FMI-Data_training feat/csc-roihu-port
```

Nothing changed locally? Skip to step 3.

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

Worth the ~5 minutes whenever you changed `SELECTED_COLUMNS` or anything in
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

```bash
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed
```

Want `COMPLETED` and `0:0`. `FAILED`, or any non-zero ExitCode, means at least
one model did not train — check that task's `.out` file.

```bash
ls /scratch/project_2019266/railway-fmi/run_*/data/output/100[0-4]-*/
```

Each directory should hold a `.joblib` model, a metrics `.json`, and PNG/PDF
figures. An empty directory means that model failed regardless of what the
terminal implied.

---

## 8. [LOCAL] Fetch the results

```bash
cd "/d/OneDrive - University of Oulu and Oamk/Railway-FMI-Data_training-CSC"
mkdir -p results
scp -r "roihu:/scratch/project_2019266/railway-fmi/run_*/data/output/100[0-4]-*" ./results/
```

Git Bash has no `rsync`, hence `scp`. The glob is `100[0-4]-*` and not `10*` on
purpose — `10*` also matches `101-preprocessed_training_ready`, which would drag
the input data back down once per run directory.

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

**Which features** — `config/const_training.py:69`, the `SELECTED_COLUMNS` list.
Names, not numbers. `'trainDelayed'` must stay: it is the target, and the
trainers split it out by name. `python main.py --dump-columns` lists everything
available.

**Which models** — edit `MODELS=(...)` in `hpc/train_array.sh:49` and match
`--array=0-N` to its length. Or run one directly:

```bash
python main.py --data-root /scratch/project_2019266/railway-fmi/run_0 --model xgboost
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
| Job hangs then `EOFError` | Config reverted to interactive | Check `SELECTED_COLUMNS` and `SCHEMA_MISMATCH_STRATEGY` |
| `SELECTED_COLUMNS references columns not in DataFrame` | Typo or a dropped column | Re-check with `--dump-columns` |
| Trainer runs but produces nothing | `--search-iterations` below 10 | Use 10 or more |
| Post-quantum SSH warning | Server lacks PQ key exchange | Cosmetic, ignore |

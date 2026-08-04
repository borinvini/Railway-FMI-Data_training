# Roihu quickstart — copy/paste commands

Real values already filled in: project `project_2019266`, user `vpozzobo`.
For the reasoning behind any of this, see `CSC-SETUP.md`.

Each block says **[LOCAL]** (your laptop, Git Bash) or **[ROIHU]** (the login
node, after `ssh roihu`). Running a block in the wrong place is the most common
way this goes wrong.

---

## Status

- [x] SSH key + certificate configured
- [x] Data staged to scratch (96 files, 25 MB)
- [x] Branch pushed to GitHub
- [ ] Environment built
- [ ] Smoke test

---

## 0. [LOCAL] Daily — before anything else

The SSH certificate expires **24 hours** after signing. When you see
`Permission denied (publickey)`, it has expired — the setup is not broken.

Re-sign at [my.csc.fi](https://my.csc.fi) → Profile → SSH PUBLIC KEYS →
three-dot menu → *Sign and download SSH certificate*, then:

```bash
mv ~/Downloads/cert.pub ~/.ssh/id_csc-cert.pub
```

Load the key once per terminal session so you stop retyping the passphrase:

```bash
eval $(ssh-agent -s) && ssh-add ~/.ssh/id_csc
```

Check the certificate's expiry any time:

```bash
ssh-keygen -L -f ~/.ssh/id_csc-cert.pub | grep Valid
```

---

## 1. [LOCAL] Push the branch — already done

`feat/csc-roihu-port` is on GitHub as of 2026-08-04. Nothing to do here unless
you make further local commits, in which case push them the same way:

```bash
cd "/d/OneDrive - University of Oulu and Oamk/Railway-FMI-Data_training-CSC"
git push Railway-FMI-Data_training feat/csc-roihu-port
```

---

## 2. [ROIHU] Connect and clone

```bash
ssh roihu
```

Confirm the project exists before spending 20 minutes on a build:

```bash
csc-workspaces
```

`project_2019266` must be listed. If it is not, stop — that is a MyCSC access
problem, not a code problem.

```bash
cd /projappl/project_2019266
git clone https://github.com/borinvini/Railway-FMI-Data_training.git railway-fmi-code
cd railway-fmi-code
git checkout feat/csc-roihu-port
```

**Already cloned before the branch was pushed?** Then `git checkout` fails with
`pathspec ... did not match any file(s) known to git` — your clone simply
predates the branch. Fetch it and retry; no need to re-clone:

```bash
cd /projappl/project_2019266/railway-fmi-code
git fetch origin
git checkout feat/csc-roihu-port
```

Verify you are on the right branch and the port files are present:

```bash
git branch --show-current          # expect: feat/csc-roihu-port
ls hpc/ environment-linux.yml      # expect: 4 scripts + the env file
```

---

## 3. [ROIHU] Build the Python environment

Takes 10–20 minutes. This is the step most likely to fail — nothing in it has
run on Linux before.

```bash
module load tykky
conda-containerize new --prefix /projappl/project_2019266/railway-env environment-linux.yml
```

Then put it on your PATH and verify every import the pipeline needs:

```bash
export PATH="/projappl/project_2019266/railway-env/bin:$PATH"
python -c "import sklearn, xgboost, lightgbm, shap, pyarrow, seaborn, haversine, matplotlib; print('env OK')"
```

Add it to your shell startup so future logins have it automatically:

```bash
echo 'export PATH="/projappl/project_2019266/railway-env/bin:$PATH"' >> ~/.bashrc
```

**If the conda solve fails**, it will name one package it cannot satisfy on
`linux-64`. Relax that single pin in `environment-linux.yml` (drop the
`=version`), rebuild, and record it under "Environment deviations" in
`CSC-SETUP.md`.

---

## 4. [ROIHU] Confirm the data arrived

```bash
ls -1 /scratch/project_2019266/railway-fmi/data/output/101-preprocessed_training_ready | wc -l
```

Expect `96`. If it is 0, re-run `bash hpc/stage_data.sh` from your laptop.

---

## 5. [ROIHU] Smoke test

One cheap job that proves the whole chain before you spend real allocation.

```bash
cd /projappl/project_2019266/railway-fmi-code
sbatch hpc/smoke_test.sh
squeue --me
```

When it finishes:

```bash
cat slurm-smoke-*.out
```

Look for **no** `EOFError`, **no** `UnicodeEncodeError`, and **no** prompt
asking you to enter column numbers. Then confirm it produced a model:

```bash
ls /scratch/project_2019266/railway-fmi/data/output/1000-xgboost_randomized_search/
seff <jobid>
```

Ignore the smoke test's CPU efficiency — 50 fits behind a serial
merge/split/SMOTE step reads low by construction. Judge efficiency on the full
run instead.

---

## 6. [ROIHU] Full run

Five models in parallel, one per array task. Recommended: a failure in one
model does not cost the other four.

```bash
sbatch hpc/train_array.sh
squeue --me
```

Or all five sequentially in a single job (also runs the SHAP analysis, but a
timeout loses everything):

```bash
sbatch hpc/train_all.sh
```

---

## 7. [LOCAL] Retrieve results

Git Bash has no `rsync`, so use `scp`:

```bash
cd "/d/OneDrive - University of Oulu and Oamk/Railway-FMI-Data_training-CSC"
mkdir -p results
scp -r "roihu:/scratch/project_2019266/railway-fmi/run_*/data/output/10*" ./results/
```

After `train_all.sh` instead (no per-task run roots):

```bash
scp -r "roihu:/scratch/project_2019266/railway-fmi/data/output/10*" ./results/
```

Scratch deletes files untouched for 180 days — copy results off.

---

## Useful while jobs run

```bash
squeue --me                      # your queue
scancel <jobid>                  # cancel a job
scancel -u vpozzobo              # cancel everything of yours
seff <jobid>                     # CPU/memory efficiency after it finishes
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
tail -f slurm-train-*.out        # follow a running job's output
csc-quota                        # disk usage against quota
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Permission denied (publickey)` | Certificate older than 24 h | Re-sign — step 0 |
| `Network is unreachable` | Wrong host | Use `roihu-cpu.csc.fi`, or the `roihu` alias |
| `sbatch: Invalid account` | Project lacks Roihu access | Check MyCSC |
| Job hangs then `EOFError` | Config reverted to interactive | Check `SELECTED_COLUMNS` and `SCHEMA_MISMATCH_STRATEGY` in `config/const_training.py` |
| `UnicodeEncodeError` in `.out` | Missing UTF-8 setting | Confirm `PYTHONIOENCODING=utf-8` in the batch script |
| `main.py: No such file` | Wrong directory | `cd /projappl/project_2019266/railway-fmi-code` |
| Trainer runs, produces nothing | `--search-iterations` below 10 | Minimum is 10 |
| Post-quantum SSH warning | Server lacks PQ key exchange | Cosmetic — CSC's to fix, safe to ignore |

# Railway-FMI-Data_training

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

Python pipeline that trains models to predict train disruptions at Oulu (station code `OL`) from weather and calendar features. Trains come from the Finnish Digitraffic open data and weather observations from the Finnish Meteorological Institute (FMI), already matched by the sibling fetcher project.

The default target, `trainDisrupted`, is true when a train arrives more than `TRAIN_DELAY_MINUTES` (5) late or is cancelled. `trainDelayed` is the pure delay label. A numeric target (`differenceInMinutes`) is also supported for XGBoost and a ridge baseline.

The code runs locally (Windows) and on the CSC Roihu supercomputer.

## Related repositories

- [Railway-FMI-Data_fetcher](https://github.com/borinvini/Railway-FMI-Data_fetcher): fetches Digitraffic and FMI data and writes the matched train+weather parquet files this project reads.

## Data

The data are not stored in this repository. Input files are `data/input/matched_data_flat_YYYY_MM.parquet`, 96 monthly files from 2018-01 to 2025-12, copied from the fetcher's `data/matched/parquet/`.

Always preprocess all months together. With `SCHEMA_MISMATCH_STRATEGY='intersect'`, a column missing from any month is silently dropped, so a subset of months can change the feature set.

Sources: [Digitraffic](https://www.digitraffic.fi/) (railway) and [FMI open data](https://en.ilmatieteenlaitos.fi/open-data) (weather).

## Method

**Preprocessing** (`src/preprocessing_pipeline.py`) works month by month and writes numbered stage folders under `data/output/` (`2-` ... `19-`, then `100-preprocessed` and `101-preprocessed_training_ready`). Notable steps:

- Weather values outside physical bounds are set to missing, never clipped.
- Calendar features (month, hour, weekday, day of month) come from the scheduled time, never the actual time, and have sin/cos encodings. Wind direction is encoded the same way.
- Cancelled trains stay in the data and are labelled disrupted.

**Training** (`src/training_pipeline.py`) runs these stages in a fixed order:

| Stage | Purpose |
|---|---|
| 500 merge_data_files | Merge the monthly files |
| 501 filter_delay_outliers | Filter delay outliers |
| 502 select_training_cols | Keep the chosen features and the target |
| 503 split_dataset | Split, then impute weather from train rows only |
| 504 balance_classes | SMOTE-Tomek (off by default) |
| 505 scale_weather_features | Scale weather features |
| 506 build_fcll_sequences | Sequence tensors for FCLL-Net |
| 700 shap_correlation_analysis | SHAP and correlation analysis |
| 701 explore_training_data | Data exploration charts and tables |
| 702 model_feature_importance | Permutation importance and ablation (off by default) |
| 999 ridge_baseline | Ridge baseline (numeric target only) |
| 1000 - 1004 | XGBoost, LightGBM, Random Forest, Logistic Regression, Naive Bayes |
| 1005 fcll_net | PyTorch FCNN + 2 LSTMs, after Huang et al. (2020) |

**Evaluation design.** The newest year is held out and never used for training or tuning. The other years are split 80/20 by operating day (never by random rows). Tabular models are tuned with `RandomizedSearchCV` over `StratifiedGroupKFold` folds grouped by day. Class imbalance is handled with class weights (`scale_pos_weight`, `class_weight`, `pos_weight`) instead of resampling.

## Installation

```sh
conda env create -f environment.yml          # Windows
conda env create -f environment-linux.yml    # Linux / CSC
```

`environment.yml` is UTF-16 LE with a BOM; keep that encoding. `pytest`, `lightgbm` and `pyarrow` are not listed in it, so install them if missing.

## Usage

Run from the repository root with the project's conda environment. `-X utf8` is required on Windows because the code prints non-ASCII symbols.

```sh
python -u -X utf8 main.py --model xgboost
```

Main flags:

| Flag | Meaning |
|---|---|
| `--model {ridge_baseline,xgboost,lightgbm,random_forest,logistic_regression,naive_bayes,fcll_net}` | One trainer plus its prerequisite stages |
| `--stages a,b,c` | Enable only these stages (cannot be combined with `--model`) |
| `--columns-file`, `--scenario`, `--list-scenarios`, `--dump-columns` | Choose and inspect the feature set |
| `--data-root` | Folder holding `data/` (also `RAILWAY_DATA_ROOT`) |
| `--n-jobs` | Parallelism (also `RAILWAY_N_JOBS`) |
| `--search-iterations` | Random search iterations, minimum 10 (also `RAILWAY_SEARCH_ITERATIONS`) |

Setup notes:

- **Features.** Copy `config/features.example.txt` to `config/features.txt` (a gitignored working copy), or generate names with `--dump-columns`. Without `--columns-file`, `main.py` uses `config/features.txt`.
- **Target.** Set `DEFAULT_TARGET_FEATURE` in `config/const_preprocessing.py`: `'trainDisrupted'` (default), `'trainDelayed'`, or `'differenceInMinutes'` (numeric). This is the only place the target is chosen.
- **Which pipeline runs.** `EXECUTE_PREPROCESSING_DATA_PIPELINE` and `EXECUTE_TRAINING_PIPELINE` in `config/const.py` select preprocessing, training, or both. Check them before each run.
- **Exploration.** `python -X utf8 main.py --stages explore_training_data` runs stage 701 alone on existing outputs.

The exit code is 1 when the run reports errors.

## Outputs

Results go to `data/output/<N>-<stage>/` (for example `1000-xgboost_randomized_search/`): metrics JSON, plots and fitted models. Logs are in `data/output/log/`. Switching target or features writes to the same folders, so copy earlier results away first.

## Running on CSC Roihu

The `hpc/` scripts submit Slurm jobs: `submit-scenarios.sh` (one task per feature scenario and model), `train_array.sh`, `train_all.sh` and `smoke_test.sh`, plus helpers for authentication, data staging and fetching results. See:

- [docs/ROIHU-QUICKSTART.md](docs/ROIHU-QUICKSTART.md)
- [docs/ROIHU-RERUN.md](docs/ROIHU-RERUN.md)
- [docs/CSC-SETUP.md](docs/CSC-SETUP.md)

## Tests

```sh
python -X utf8 -m pytest tests
```

Run from the repository root. Each past bug has a regression test whose module docstring explains it. Two tests that need real data skip when `data/input` has no parquet files.

## Repository layout

```
config/    constants, state machines, feature-file parser, example feature files
src/       preprocessing, training, exploration and FCLL-Net code
hpc/       Slurm and CSC Roihu helper scripts
scripts/   stand-alone checks (FCLL feasibility)
tests/     pytest suite
docs/      CSC runbooks and notes
main.py    command-line entry point
```

## Citation

<!-- TODO(R2.13b/c): add the article reference here once published. -->
If you use this code, please cite the accompanying article: *(reference to be added)*.

## License

This project is released under the GNU General Public License version 3 (GPL-3.0); see [LICENSE](LICENSE). The companion fetcher repository is released under the same licence.

<!-- TODO(R2.13b/c): once both repositories have a GPL-3.0 licence file, a tagged release and a DOI, replace the placeholders below. -->
The version used for the article is archived on Zenodo with the DOI `<DOI>` (release `<tag>`).

## Acknowledgements

Computing resources were provided by CSC - IT Center for Science, Finland (project `project_2019266`). Railway data come from Fintraffic's Digitraffic service and weather data from the Finnish Meteorological Institute, both open data. Check their licence terms (Creative Commons Attribution 4.0) before redistributing data.

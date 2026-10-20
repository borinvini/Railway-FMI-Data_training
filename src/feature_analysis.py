"""Feature-level diagnostics behind stage 701 (feature_importance_analysis).

Three questions SHAP on the train rows cannot answer:

* which feature groups still matter on rows the model never saw (grouped permutation
  importance on the test and hold-out sets),
* what the model loses when a group is removed altogether (drop-group ablation),
* which continuous columns carry the same ranking information (Spearman clustering).

Everything works on groups, not columns: a sin/cos pair, a one-hot family of weather
scenarios or the wawa groups mean nothing as a single column, so they are shuffled,
dropped and kept together. Pure functions, no pipeline state, so the tests can drive
them with small synthetic frames.
"""
import re

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import get_scorer, mean_absolute_error, mean_squared_error, r2_score

from src.cv_utils import to_minutes

_SCENARIO_FAMILY = re.compile(r"^weather_scenario(?:_(\d+h))?_")


def feature_groups(columns):
    """{group name: [columns]} with encoding families kept whole.

    `<stem>_sin` + `<stem>_cos` is one group `<stem>` (month, hour, day_week, day_month,
    wind_dir); `weather_scenario[_12h|_24h|_72h]_*` is one group per window; `wawa_group_*`
    is one group. A `_sin` without its `_cos` and every other column is its own group.
    Order follows the first appearance of each group in `columns`.
    """
    columns = list(columns)
    present = set(columns)
    groups = {}
    for col in columns:
        if col.endswith("_sin") and col[:-4] + "_cos" in present:
            name = col[:-4]
        elif col.endswith("_cos") and col[:-4] + "_sin" in present:
            name = col[:-4]
        elif (match := _SCENARIO_FAMILY.match(col)):
            name = "weather_scenario" + (f"_{match.group(1)}" if match.group(1) else "")
        elif col.startswith("wawa_group_"):
            name = "wawa_group"
        else:
            name = col
        groups.setdefault(name, []).append(col)
    return groups


def protected_columns(groups):
    """Columns that belong to a multi-column family: never clustered, never split."""
    return {col for cols in groups.values() if len(cols) > 1 for col in cols}


def make_score_fn(is_classification, metric, y_shift):
    """f(model, X, y) -> score, higher is better, for a model fitted on `y` or on its log.

    Numeric targets are scored in minutes: predictions go through `cv_utils.to_minutes`
    (y_shift None = the model already predicts minutes), so the score is -MAE / -RMSE / R2
    of the delay itself. Classification uses the sklearn scorer named by `metric`.
    """
    if is_classification:
        scorer = get_scorer(metric)
        return lambda model, X, y: float(scorer(model, X, y))

    def score(model, X, y):
        pred = to_minutes(model.predict(X), y_shift)
        if metric == "mae":
            return -float(mean_absolute_error(y, pred))
        if metric == "rmse":
            return -float(np.sqrt(mean_squared_error(y, pred)))
        if metric == "r2":
            return float(r2_score(y, pred))
        raise ValueError(f"unknown regression metric {metric!r}")

    return score


def fit_xgboost(X, y_fit, seed, *, is_classification, objective, n_jobs, params,
                scale_pos_weight=None):
    """The plain model every analysis refits: defaults plus `params`, no search."""
    if is_classification:
        model = xgb.XGBClassifier(random_state=seed, n_jobs=n_jobs,
                                  scale_pos_weight=scale_pos_weight or 1.0, **params)
    else:
        model = xgb.XGBRegressor(random_state=seed, n_jobs=n_jobs, objective=objective, **params)
    return model.fit(X, y_fit)


def grouped_permutation_importance(model, X, y, groups, score_fn, n_repeats, seed, noise_factor=2.0):
    """Score drop when a group is shuffled across the rows of X.

    One row order is drawn per repeat and applied to every column of the group, so a
    sin/cos pair stays a consistent pair of values (just attached to the wrong row).
    Importance = baseline score - shuffled score, so a positive value means the model
    uses the group. `within_noise` is True when the mean is not clearly away from zero:
    |mean| <= `noise_factor` * the standard deviation over the repeats.
    """
    rng = np.random.default_rng(seed)
    baseline = score_fn(model, X, y)
    work = X.copy()
    rows = []
    for name, cols in groups.items():
        originals = {c: X[c].to_numpy() for c in cols}
        drops = []
        for _ in range(n_repeats):
            order = rng.permutation(len(X))
            for c in cols:
                work[c] = originals[c][order]
            drops.append(baseline - score_fn(model, work, y))
        for c in cols:
            work[c] = originals[c]
        mean = float(np.mean(drops))
        std = float(np.std(drops, ddof=1)) if n_repeats > 1 else float("nan")
        rows.append({"group": name, "columns": " | ".join(cols), "n_columns": len(cols),
                     "baseline_score": baseline, "importance_mean": mean, "importance_std": std,
                     "within_noise": bool(np.isnan(std) or abs(mean) <= noise_factor * std)})
    out = pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def drop_group_ablation(fit_fn, X_train, y_train_fit, evals, groups, extra_sets, score_fn,
                        noise_seeds, seed, noise_factor=2.0):
    """Refit without each group (and each named set) and score the rest on every split.

    `fit_fn(X, y_fit, seed)` returns a fitted model; `evals` is {split: (X, y_minutes)}.
    The full model is fitted with `noise_seeds` seeds first: its spread per split is the
    noise floor, and a change within `noise_factor` times that spread is flagged
    `within_noise`. change = baseline score - score without the group, so positive
    means the group helps. Returns (table, baseline per split, noise std per split).
    """
    seeds = [seed + k for k in range(max(1, noise_seeds))]
    per_seed = {split: [] for split in evals}
    for s in seeds:
        full = fit_fn(X_train, y_train_fit, s)
        for split, (X_eval, y_eval) in evals.items():
            per_seed[split].append(score_fn(full, X_eval[X_train.columns], y_eval))
    baseline = {split: float(scores[0]) for split, scores in per_seed.items()}
    noise = {split: (float(np.std(scores, ddof=1)) if len(scores) > 1 else float("nan"))
             for split, scores in per_seed.items()}

    candidates = [(name, "group", list(cols)) for name, cols in groups.items()]
    seen = {tuple(cols) for _, _, cols in candidates}
    for name, wanted in (extra_sets or {}).items():
        cols = [c for c in wanted if c in X_train.columns]
        if cols and tuple(cols) not in seen:
            candidates.append((name, "set", cols))
            seen.add(tuple(cols))

    rows = []
    for name, kind, cols in candidates:
        keep = [c for c in X_train.columns if c not in cols]
        if not keep:
            continue
        model = fit_fn(X_train[keep], y_train_fit, seed)
        row = {"dropped": name, "kind": kind, "columns": " | ".join(cols), "n_columns": len(cols)}
        for split, (X_eval, y_eval) in evals.items():
            score = score_fn(model, X_eval[keep], y_eval)
            change = baseline[split] - score
            row[f"{split}_score"] = score
            row[f"{split}_change"] = change
            row[f"{split}_within_noise"] = bool(np.isnan(noise[split])
                                                 or abs(change) <= noise_factor * noise[split])
        rows.append(row)
    return pd.DataFrame(rows), baseline, noise


def spearman_clusters(X, protected, threshold):
    """Cluster the continuous columns by Spearman correlation and pick one per cluster.

    Columns in `protected` (encoding families) are left out of the clustering and are
    always kept, so the suggestion can never keep one half of a sin/cos pair. Distance
    is 1 - |rho|, average linkage, cut at 1 - `threshold`: two columns end in one cluster
    when they are, on average, at least `threshold` correlated. The representative is the
    member with the highest mean |rho| to the rest of its cluster (ties: first by name).

    Returns a dict: `corr` (Spearman matrix of the clustered columns), `clusters` (one row
    per clustered column), `linkage` and `labels` (for a dendrogram, None with fewer than
    two columns), `constant_columns` (no variance, so no correlation; kept out of the
    suggestion), `protected_kept` and `suggested` (X's column order).
    """
    protected = set(protected)
    candidates = [c for c in X.columns if c not in protected]
    constant = [c for c in candidates if X[c].nunique(dropna=True) <= 1]
    cols = [c for c in candidates if c not in constant]

    clusters, Z, corr = pd.DataFrame(), None, pd.DataFrame(index=cols, columns=cols, dtype=float)
    if cols:
        corr = X[cols].rank().corr()
        if len(cols) == 1:
            labels = np.array([1])
        else:
            dist = 1.0 - corr.abs().to_numpy()
            dist = np.nan_to_num((dist + dist.T) / 2.0, nan=1.0)
            np.fill_diagonal(dist, 0.0)
            Z = linkage(squareform(np.clip(dist, 0.0, None), checks=False), method="average")
            labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

        rows = []
        for label in sorted(set(labels)):
            members = sorted(c for c, l in zip(cols, labels) if l == label)
            if len(members) == 1:
                mean_abs = {members[0]: float("nan")}
            else:
                sub = corr.loc[members, members].abs().to_numpy()
                np.fill_diagonal(sub, np.nan)
                mean_abs = dict(zip(members, np.nanmean(sub, axis=1)))
            rep = members[0] if len(members) == 1 else max(members, key=lambda m: (mean_abs[m], -members.index(m)))
            for m in members:
                rows.append({"feature": m, "cluster": int(label), "cluster_size": len(members),
                             "representative": m == rep, "mean_abs_corr_in_cluster": mean_abs[m]})
        clusters = pd.DataFrame(rows)

    chosen = set(clusters.loc[clusters["representative"], "feature"]) if len(clusters) else set()
    protected_kept = [c for c in X.columns if c in protected]
    suggested = [c for c in X.columns if c in protected or c in chosen]
    return {"corr": corr, "clusters": clusters, "linkage": Z, "labels": cols if Z is not None else None,
            "constant_columns": constant, "protected_kept": protected_kept, "suggested": suggested}

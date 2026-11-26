"""Scenario x model table from a fetched and sorted scenario grid.

Usage (repo root):  python -X utf8 hpc/grid_summary.py <results_dir>
Reads <results_dir>/<N - scenario name>/{999,100[0-3]}-*/<model>*.json and writes
<results_dir>/grid_summary.csv. hpc/fetch-stations.sh calls it once per station.

Three kinds of grid, told apart by the JSON:
- numeric (problem_type "regression", e.g. differenceInMinutes): hold-out MAE,
  mae_skill (1 - MAE / MAE of always predicting the train median), RMSE and R2.
  ridge_baseline (999) is the reference the tree models are read against.
- binary (trainDisrupted): ROC-AUC, PR-AUC lift over the base rate, F1 against
  f1_always_positive.
- 4-class (target_type "multiclass"): one-vs-rest macro AUC, macro F1 against
  f1_macro_majority, and the 4x4 confusion matrix collapsed to on time vs late.
A grid mixing kinds is refused: one fetch is one grid.
"""
import csv
import glob
import json
import os
import re
import sys

MODELS = {"999": "ridge", "1000": "xgb", "1001": "lgbm", "1002": "rf", "1003": "logreg"}
ORDER = ("ridge", "xgb", "lgbm", "rf", "logreg")
TOP_CLASS = "late_15_plus_or_cancelled"


def _result_files(scenario_dir):
    """(code, path) of each model's result JSON in one scenario folder."""
    found = []
    for path in glob.glob(os.path.join(scenario_dir, "100[0-3]-*", "*iteration_analysis*.json")):
        found.append((os.path.basename(os.path.dirname(path)).split("-")[0], path))
    for path in glob.glob(os.path.join(scenario_dir, "999-*", "ridge_baseline_*.json")):
        found.append(("999", path))
    return found


def collapsed_binary(cm):
    """(precision, recall, F1, accuracy, positive rate) of classes 1-3 vs class 0."""
    tn = cm[0][0]
    fp = sum(cm[0][1:])
    fn = sum(row[0] for row in cm[1:])
    tp = sum(sum(row[1:]) for row in cm[1:])
    n = tn + fp + fn + tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1, (tn + tp) / n, (tp + fn) / n


def _kind(d):
    if d.get("problem_type") == "regression" or "test_mae" in d.get("final_metrics", {}):
        return "numeric"
    return "multiclass" if d.get("target_type") == "multiclass" else "binary"


def _row(scenario, model, d):
    t, h = d["final_metrics"], d.get("holdout_metrics", {})
    info = d.get("dataset_info", {})
    row = dict(s=scenario, model=model, kind=_kind(d), nf=info.get("features_count"),
               n=info.get("holdout_samples"), best=d.get("best_iteration", d.get("best_alpha")))
    if row["kind"] == "numeric":
        row.update(
            t_mae=t["test_mae"], t_skill=t.get("mae_skill"), t_rmse=t["test_rmse"], t_r2=t["test_r2"],
            h_mae=h.get("mae"), h_skill=h.get("mae_skill"), h_rmse=h.get("rmse"), h_r2=h.get("r2"),
            h_base_mae=h.get("baseline_mae"))
    elif row["kind"] == "multiclass":
        _, _, t_bf1, _, _ = collapsed_binary(t["confusion_matrix"])
        h_bp, h_br, h_bf1, h_bacc, h_bpos = collapsed_binary(h["confusion_matrix"])
        row.update(
            t_auc=t["test_auc"], t_pr=t["test_pr_auc"], h_auc=h["auc"], h_pr=h["pr_auc"],
            t_f1=t["test_f1"], t_f1_major=t["f1_macro_majority"], t_bal=t["balanced_accuracy"],
            t_top_f1=t["per_class"][TOP_CLASS]["f1"],
            h_f1=h["f1"], h_f1_major=h["f1_macro_majority"], h_bal=h["balanced_accuracy"],
            h_top_f1=h["per_class"][TOP_CLASS]["f1"], h_within1=h.get("within_one_class_accuracy"),
            t_bin_f1=t_bf1, h_bin_prec=h_bp, h_bin_rec=h_br, h_bin_f1=h_bf1, h_bin_acc=h_bacc,
            h_bin_f1base=2 * h_bpos / (1 + h_bpos))
    else:
        row.update(
            t_auc=t["test_auc"], t_pr=t["test_pr_auc"], h_auc=h["auc"], h_pr=h["pr_auc"],
            t_base=t["positive_rate_test"], h_base=h["positive_rate"],
            h_prec=h["precision"], h_rec=h["recall"], h_f1=h["f1"], h_f1base=h["f1_always_positive"])
    return row


def collect_rows(root):
    """One row per scenario x model. Raises ValueError for an empty or mixed grid."""
    rows = []
    for sdir in glob.glob(os.path.join(root, "*", "")):
        m = re.match(r"(\d+) - ", os.path.basename(os.path.dirname(sdir)))
        if not m:
            continue
        for code, path in _result_files(sdir):
            with open(path, encoding="utf-8") as fh:
                rows.append(_row(int(m.group(1)), MODELS[code], json.load(fh)))
    if not rows:
        raise ValueError(f"no results found under {root}")
    kinds = {r["kind"] for r in rows}
    if len(kinds) > 1:
        raise ValueError(f"results mix {sorted(kinds)} cells; fetch one grid at a time")
    rows.sort(key=lambda r: (r["s"], ORDER.index(r["model"])))
    return rows


def _fmt(v, spec):
    return format(v, spec) if isinstance(v, (int, float)) else "-"


def print_tables(rows, out=print):
    kind = rows[0]["kind"]
    models = [m for m in ORDER if any(r["model"] == m for r in rows)]
    scenarios = sorted({r["s"] for r in rows})
    out(f"{len(rows)} cells, {len(scenarios)} scenarios x {len(models)} models - {kind} target")

    def table(title, fmt, width=14):
        out(f"\n{title}")
        out("scen | " + "".join(f"{m:<{width}}" for m in models))
        for s in scenarios:
            rs = {r["model"]: r for r in rows if r["s"] == s}
            out(f"{s:>4} | " + "".join(f"{(fmt(rs[m]) if m in rs else '-'):<{width}}" for m in models))

    if kind == "numeric":
        table("HOLD-OUT MAE in minutes (test in brackets)",
              lambda r: f"{_fmt(r['h_mae'], '.2f')} ({_fmt(r['t_mae'], '.2f')})")
        table("HOLD-OUT mae_skill vs the train median, > 0 beats it (test in brackets)",
              lambda r: f"{_fmt(r['h_skill'], '+.3f')} ({_fmt(r['t_skill'], '+.3f')})", 17)
        table("HOLD-OUT RMSE / R2", lambda r: f"{_fmt(r['h_rmse'], '.1f')}/{_fmt(r['h_r2'], '.3f')}")
        base = next((r["h_base_mae"] for r in rows if r.get("h_base_mae") is not None), None)
        if base is not None:
            out(f"\nhold-out MAE of always predicting the train median: {base:.2f} min")
    elif kind == "multiclass":
        table("HOLD-OUT macro ROC-AUC (test in brackets)",
              lambda r: f"{r['h_auc']:.3f} ({r['t_auc']:.3f})")
        table("HOLD-OUT macro F1 / majority-class F1",
              lambda r: f"{r['h_f1']:.3f}/{r['h_f1_major']:.3f}")
        table("HOLD-OUT balanced accuracy / F1 of the top class",
              lambda r: f"{r['h_bal']:.3f}/{r['h_top_f1']:.3f}")
    else:
        table("HOLD-OUT ROC-AUC (test in brackets)", lambda r: f"{r['h_auc']:.3f} ({r['t_auc']:.3f})")
        table("HOLD-OUT PR-AUC lift over the base rate", lambda r: f"{r['h_pr'] - r['h_base']:+.3f}")
        table("HOLD-OUT F1 vs always-disrupted F1",
              lambda r: f"{r['h_f1']:.3f}/{r['h_f1base']:.3f}")


def write_csv(rows, root):
    fieldnames = []
    for r in rows:
        fieldnames += [k for k in r if k not in fieldnames]
    path = os.path.join(root, "grid_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = argv[0] if argv else "results"
    try:
        rows = collect_rows(root)
    except ValueError as exc:
        print(f"grid_summary: {exc}", file=sys.stderr)
        return 1
    print_tables(rows)
    print("\nwrote", write_csv(rows, root))
    return 0


if __name__ == "__main__":
    sys.exit(main())

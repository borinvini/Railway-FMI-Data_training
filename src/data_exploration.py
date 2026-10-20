"""Model-free statistics behind stage 701 (explore_training_data).

Describes the data itself: how the target is distributed and moves over time, how it
changes with the hour, the weekday, the month, the wind direction and each weather
reading, how the features relate to each other, and which of them look different in the
newest (hold-out) year. Only counts, ranks, bins, correlations and distribution tests;
no model is fitted anywhere in this module or in src/exploration_plots.py.

Most calendar and wind features reach the pipeline as sin/cos pairs, which are right for
a model and unreadable for a person. `recover_readable_variables` turns each pair back
into the hour, month, weekday, day of month or wind direction it encodes (exactly: the
round-trip error is at floating-point level), so every chart can use real units.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

TWO_PI = 2.0 * np.pi

# stem of the sin/cos pair -> the variable it encodes. `first` is the value of the first
# position on the circle (preprocessing uses (value - first) / period for the angle).
CIRCULAR_SPECS = {
    "hour": {"name": "hour", "period": 24.0, "integer": False, "first": 0},
    "month": {"name": "month", "period": 12.0, "integer": True, "first": 1},
    "day_week": {"name": "weekday", "period": 7.0, "integer": True, "first": 1},   # 1 = Sunday
    "day_month": {"name": "day_of_month", "period": 31.0, "integer": True, "first": 1},
    "wind_dir": {"name": "wind_direction", "period": 360.0, "integer": False, "first": 0},
}
WIND_COLUMN = "wind_direction"
WIND_CALM_COLUMN = "wind_calm"
COMPASS_16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
              "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
WEEKDAY_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]   # weekday 1..7
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def recover_readable_variables(features):
    """(readable frame, meta) with one column per variable a person would recognise.

    A `<stem>_sin` / `<stem>_cos` pair of a known stem becomes the hour (decimal, 0-24),
    month (1-12), weekday (1 = Sunday), day of month (1-31) or wind direction (degrees);
    calm wind (sin = cos = 0) becomes NaN in `wind_direction` plus True in `wind_calm`.
    Every other numeric column passes through under its own name. Columns with no
    variation are left out and listed in meta["__constant__"].

    meta[name] = {"kind": "circular" | "binary" | "continuous", "sources": [...]}.
    """
    readable, meta, constant, done = {}, {}, [], set()
    for col in features.columns:
        if col in done:
            continue
        stem = col[:-4] if col.endswith(("_sin", "_cos")) else None
        spec = CIRCULAR_SPECS.get(stem)
        sin_col, cos_col = (f"{stem}_sin", f"{stem}_cos") if stem else (None, None)
        if spec and sin_col in features.columns and cos_col in features.columns:
            s = pd.to_numeric(features[sin_col], errors="coerce")
            c = pd.to_numeric(features[cos_col], errors="coerce")
            value = np.mod(np.arctan2(s, c), TWO_PI) * spec["period"] / TWO_PI
            if spec["integer"]:
                value = np.mod(np.round(value), spec["period"]) + spec["first"]
            if spec["name"] == WIND_COLUMN:
                calm = (s == 0) & (c == 0)
                value = value.mask(calm)
                readable[WIND_CALM_COLUMN] = calm
            readable[spec["name"]] = value
            meta[spec["name"]] = {"kind": "circular", "sources": [sin_col, cos_col]}
            done.update((sin_col, cos_col))
            continue
        if not pd.api.types.is_numeric_dtype(features[col]):
            continue
        values = features[col].astype(float) if features[col].dtype == bool else features[col]
        if values.nunique(dropna=True) <= 1:
            constant.append(col)
            continue
        readable[col] = values
        meta[col] = {"kind": "binary" if values.nunique(dropna=True) <= 2 else "continuous",
                     "sources": [col]}
    meta["__constant__"] = constant
    return pd.DataFrame(readable, index=features.index), meta


def variable_names(meta):
    return [name for name in meta if not name.startswith("__")]


def wind_sector_labels(n_sectors):
    return COMPASS_16 if n_sectors == 16 else [f"{i * 360 / n_sectors:.0f}°" for i in range(n_sectors)]


def bin_codes(readable, meta, n_bins, n_sectors):
    """One categorical code per variable: the natural unit of the variable, or quantile bins.

    hour -> 0..23, month -> 1..12, weekday -> 1..7, day of month -> 1..31, wind direction ->
    compass sector 0..n-1 (calm = -1), binary -> its value, continuous -> `n_bins` quantile
    bins (fewer when the variable has many ties). NaN stays NaN.
    """
    codes = {}
    for name in variable_names(meta):
        x = readable[name]
        if name == "hour":
            codes[name] = np.floor(x)
        elif meta[name]["kind"] == "circular" and name != WIND_COLUMN:
            codes[name] = x
        elif name == WIND_COLUMN:
            width = 360.0 / n_sectors
            sector = np.floor(np.mod(x + width / 2.0, 360.0) / width)
            calm = readable[WIND_CALM_COLUMN]
            codes[name] = sector.mask(calm, -1.0)
        elif meta[name]["kind"] == "binary":
            codes[name] = x
        else:
            codes[name] = _quantile_codes(x, n_bins)
    return pd.DataFrame(codes, index=readable.index)


def _quantile_codes(x, n_bins):
    """Quantile bins that give every heavily repeated value its own bin.

    Precipitation is zero for 93% of the trains and snow depth is -1 for 32% and 0 for 20%;
    a plain quantile cut puts all of that into one bin and loses the variable. A value that
    holds at least 1/n_bins of the rows becomes a bin of its own (coded 0, 1, ... in value
    order) and the remaining rows are cut into the rest of the bins.
    """
    counts = x.value_counts(normalize=True)
    heavy = sorted(counts[counts >= 1.0 / n_bins].index)
    out = pd.Series(np.nan, index=x.index)
    for i, value in enumerate(heavy):
        out[x == value] = float(i)
    rest = x.where(~x.isin(heavy))
    if rest.notna().sum() > 1 and rest.nunique() > 1:
        out[rest.notna()] = pd.qcut(rest.dropna(), max(2, n_bins - len(heavy)), labels=False,
                                    duplicates="drop").astype(float) + len(heavy)
    elif rest.notna().any():
        out[rest.notna()] = float(len(heavy))
    return out


def rank_eta_squared(bins, y):
    """Share of the target's rank variance explained by `bins` (eta squared on ranks).

    0 = the groups look alike; close to 1 = the group nearly determines the rank of the target
    (the ceiling is about 1 - 1/k^2 for k groups, since ranks still vary inside a group). Ranks, not
    values, because the delay is heavy-tailed; groups, not a line, so a circular variable
    (hour, wind direction) or a binary flag is measured on the same scale as a weather
    reading. It is marginal (one variable at a time) and rises by about (k - 1) / (n - 1)
    from chance alone, which is returned as `chance`.
    """
    d = pd.DataFrame({"b": np.asarray(bins, dtype=float), "r": np.asarray(y, dtype=float)}).dropna()
    n = len(d)
    out = {"eta2": np.nan, "n": n, "k": 0, "chance": np.nan}
    if n < 3:
        return out
    ranks = d["r"].rank().to_numpy()
    grand = ranks.mean()
    ss_total = float(((ranks - grand) ** 2).sum())
    groups = pd.DataFrame({"b": d["b"].to_numpy(), "r": ranks}).groupby("b")["r"].agg(["size", "mean"])
    out["k"] = len(groups)
    if ss_total == 0 or len(groups) < 2:
        return out
    out["eta2"] = float((groups["size"] * (groups["mean"] - grand) ** 2).sum() / ss_total)
    out["chance"] = (len(groups) - 1) / (n - 1)
    return out


def association_table(readable, meta, codes, y):
    """Per variable: the rank eta squared with the target and, where a line makes sense, Spearman rho."""
    rows = []
    for name in variable_names(meta):
        stat = rank_eta_squared(codes[name], y)
        rho = np.nan
        if meta[name]["kind"] != "circular":
            pair = pd.DataFrame({"x": readable[name], "y": y}).dropna()
            if len(pair) > 2 and pair["x"].nunique() > 1:
                rho = float(stats.spearmanr(pair["x"], pair["y"])[0])
        rows.append({"variable": name, "kind": meta[name]["kind"], "n": stat["n"], "n_groups": stat["k"],
                     "eta2": stat["eta2"], "chance_level": stat["chance"], "spearman_rho": rho})
    return pd.DataFrame(rows).sort_values("eta2", ascending=False, na_position="last").reset_index(drop=True)


def association_by_year(codes, y, years):
    """variable x year of the rank eta squared: is what explains the target stable over time?"""
    out = {}
    for year in sorted(pd.unique(years.dropna())):
        mask = (years == year).to_numpy()
        out[int(year)] = {name: rank_eta_squared(codes.loc[mask, name], y[mask])["eta2"] for name in codes.columns}
    return pd.DataFrame(out)


def drift_table(readable, meta, hold_mask):
    """How the newest year differs from the earlier years, per variable (two-sample KS).

    `ks` is the largest gap between the two cumulative distributions (0 = identical,
    1 = disjoint); a p-value would be useless at this sample size. `median_shift_iqr` is the
    shift of the median in units of the earlier years' interquartile range (NaN for
    circular variables, whose median is meaningless).
    """
    hold_mask = np.asarray(hold_mask, dtype=bool)
    rows = []
    for name in variable_names(meta):
        x = readable[name]
        dev, hold = x[~hold_mask].dropna(), x[hold_mask].dropna()
        if len(dev) < 2 or len(hold) < 2:
            continue
        iqr = float(dev.quantile(0.75) - dev.quantile(0.25))
        shift = np.nan
        if meta[name]["kind"] != "circular" and iqr > 0:
            shift = float((hold.median() - dev.median()) / iqr)
        rows.append({"variable": name, "kind": meta[name]["kind"], "ks": float(stats.ks_2samp(dev, hold)[0]),
                     "median_development": float(dev.median()), "median_holdout": float(hold.median()),
                     "median_shift_iqr": shift})
    return pd.DataFrame(rows).sort_values("ks", ascending=False).reset_index(drop=True)


def _wilson(hits, n, z=1.96):
    """95% Wilson score interval of a share (well behaved for small n and shares near 0 or 1)."""
    p = hits / n
    denom = 1.0 + z ** 2 / n
    centre = (p + z ** 2 / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z ** 2 / (4.0 * n ** 2)) / denom
    return centre - half, centre + half


def profile_by(keys, y, is_binary, threshold=None):
    """Target summary per group.

    Always: `n`, `share` (the share of trains that are late, i.e. delay above `threshold`, or
    the positive rate of a binary target) with its 95% Wilson band `share_low`/`share_high`.
    Numeric target, additionally: `center` (median), `low`/`high` (25th/75th percentile),
    `p90` and `mean`, in the unit of the target. The median alone is a poor summary of a
    delay in whole minutes (it is 1 for almost every group), which is why `share` leads.
    """
    d = pd.DataFrame({"key": np.asarray(keys, dtype=float), "y": np.asarray(y, dtype=float)}).dropna()
    if is_binary:
        d["hit"] = d["y"]
    else:
        if threshold is None:
            raise ValueError("a numeric target needs the late threshold")
        d["hit"] = (d["y"] > threshold).astype(float)
    grouped = d.groupby("key")
    out = pd.DataFrame({"n": grouped.size(), "hits": grouped["hit"].sum()})
    out["share"] = out["hits"] / out["n"]
    out["share_low"], out["share_high"] = _wilson(out["hits"], out["n"])
    if is_binary:
        out["center"], out["low"], out["high"] = out["share"], out["share_low"], out["share_high"]
        out["p90"] = out["mean"] = np.nan
    else:
        gy = grouped["y"]
        out["center"], out["low"], out["high"] = gy.median(), gy.quantile(0.25), gy.quantile(0.75)
        out["p90"], out["mean"] = gy.quantile(0.90), gy.mean()
    return out.drop(columns="hits").reset_index()


def monthly_series(year_month, y, is_binary, threshold):
    """Per calendar month (YYYY-MM): rows, centre, p90 and the share above `threshold`."""
    d = pd.DataFrame({"ym": year_month, "y": np.asarray(y, dtype=float)}).dropna()
    grouped = d.groupby("ym")["y"]
    out = pd.DataFrame({"n": grouped.size()})
    out["center"] = grouped.mean() if is_binary else grouped.median()
    out["p90"] = np.nan if is_binary else grouped.quantile(0.90)
    out["share_above"] = grouped.mean() if is_binary else grouped.apply(lambda s: float((s > threshold).mean()))
    out.index = pd.to_datetime(out.index.astype(str) + "-01")
    return out.sort_index()


def data_summary(frame):
    """One row per numeric column: count, missing, spread and the shares of zeros and of -1."""
    rows = []
    for col in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        x = frame[col]
        valid = x.dropna()
        rows.append({"column": col, "n": int(valid.size), "missing": int(x.isna().sum()),
                     "min": valid.min(), "p1": valid.quantile(0.01), "median": valid.median(),
                     "p99": valid.quantile(0.99), "max": valid.max(), "mean": valid.mean(),
                     "share_zero": float((valid == 0).mean()), "share_minus_one": float((valid == -1).mean())})
    return pd.DataFrame(rows)


def spearman_matrix(X):
    """Spearman correlation of the numeric columns, ordered so that similar columns are adjacent.

    Returns (matrix in leaf order, the linkage). Columns without variation are left out.
    """
    cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique(dropna=True) > 1]
    corr = X[cols].rank().corr()
    if len(cols) < 3:
        return corr, None
    dist = np.nan_to_num(1.0 - corr.abs().to_numpy(), nan=1.0)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(np.clip(dist, 0.0, None), checks=False), method="average")
    order = leaves_list(Z)
    return corr.iloc[order, order], Z

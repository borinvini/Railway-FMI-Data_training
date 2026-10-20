"""Charts for stage 701 (explore_training_data). Matplotlib only; no model.

Every function takes already-computed tables from src/data_exploration.py and returns a
figure (or None when the data the chart needs is not there, e.g. no wind direction in the
feature set). Each figure carries a one-line "how to read this" caption, because the point
of the stage is to explain the data to a person. Development years are blue, the hold-out
year is orange.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram

from src.data_exploration import MONTH_NAMES, WEEKDAY_NAMES

DEV, HOLD = "#1f77b4", "#ff7f0e"
KIND_COLORS = {"circular": "tab:purple", "binary": "tab:green", "continuous": "tab:blue"}


def _caption(fig, text):
    fig.text(0.01, 0.008, text, fontsize=8, color="dimgray", ha="left", va="bottom", wrap=True)


def _finish(fig, caption, top=1.0):
    _caption(fig, caption)
    fig.tight_layout(rect=(0, 0.05, 1, top))
    return fig


def _grid(n, ncols=4, cell=(3.1, 2.6), **kw):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell[0] * ncols, cell[1] * nrows + 0.5), squeeze=False, **kw)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat[:n]


# --- 01 target distribution -------------------------------------------------------

def plot_target_distribution(y_dev, y_hold, is_binary, threshold, label):
    y_dev, y_hold = np.asarray(y_dev, float), np.asarray(y_hold, float)
    y_dev, y_hold = y_dev[~np.isnan(y_dev)], y_hold[~np.isnan(y_hold)]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    if is_binary:
        groups = [("development years", y_dev, DEV)] + ([("hold-out year", y_hold, HOLD)] if len(y_hold) else [])
        rates = [g[1].mean() for g in groups]
        ax.bar([g[0] for g in groups], rates, color=[g[2] for g in groups])
        for i, (name, y, _) in enumerate(groups):
            ax.text(i, rates[i], f"{rates[i]:.1%}\nn = {len(y):,}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, max(rates) * 1.25)
        ax.set_ylabel(f"share of trains: {label}")
        ax.set_title("Class balance of the target")
        return _finish(fig, "How to read: each bar is the share of trains that were late or cancelled. "
                            "Similar bars mean the target behaves the same in the hold-out year.")
    pooled = np.concatenate([y_dev, y_hold])
    lo, hi = np.percentile(pooled, [1, 99])
    bins = np.linspace(lo, hi, 61)
    ax.hist(y_dev, bins=bins, density=True, histtype="stepfilled", alpha=0.45, color=DEV, label="development years")
    if len(y_hold):
        ax.hist(y_hold, bins=bins, density=True, histtype="step", linewidth=2, color=HOLD, label="hold-out year")
    ax.set_yscale("log")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1, label=f"{threshold:g}-minute threshold")
    lines = [f"development: median {np.median(y_dev):.1f}, mean {y_dev.mean():.1f}, above threshold {(y_dev > threshold).mean():.0%}"]
    if len(y_hold):
        lines.append(f"hold-out: median {np.median(y_hold):.1f}, mean {y_hold.mean():.1f}, above threshold {(y_hold > threshold).mean():.0%}")
    ax.text(0.98, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "lightgray"})
    ax.set_xlabel(f"{label}; axis trimmed to the 1st-99th percentile")
    ax.set_ylabel("density (log scale)")
    ax.set_title("Distribution of the target")
    ax.legend(loc="center right", fontsize=8)
    return _finish(fig, "How to read: most trains are within a few minutes of schedule; the long right tail holds the big "
                        "delays. Both histograms are density-normalised so the two periods compare in shape, not in size.")


# --- 02 target over time ----------------------------------------------------------

def plot_target_over_time(series, is_binary, threshold, hold_year, label):
    if len(series) < 3:
        return None
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    top.plot(series.index, series["center"], color=DEV, linewidth=1.5, label="rate" if is_binary else "median")
    if not is_binary:
        top.plot(series.index, series["p90"], color=DEV, linewidth=1.2, linestyle="--", label="90th percentile")
    top.set_ylabel("share of trains" if is_binary else f"{label}")
    top.legend(fontsize=8, loc="upper left")
    top.set_title("Target month by month")
    if not is_binary:
        bottom.plot(series.index, series["share_above"], color="tab:red", linewidth=1.3)
        bottom.set_ylabel(f"share above {threshold:g} min")
    else:
        bottom.plot(series.index, series["center"], color="tab:red", linewidth=1.3)
        bottom.set_ylabel("rate")
    bottom.set_ylim(bottom=0)
    count_ax = bottom.twinx()
    count_ax.bar(series.index, series["n"], width=22, color="lightgray", alpha=0.6)
    count_ax.set_ylabel("trains per month", color="gray")
    count_ax.set_zorder(0)
    bottom.set_zorder(1)
    bottom.patch.set_visible(False)
    if hold_year is not None:
        for ax in (top, bottom):
            ax.axvspan(pd.Timestamp(int(hold_year), 1, 1), pd.Timestamp(int(hold_year), 12, 31), color=HOLD, alpha=0.12)
        top.text(pd.Timestamp(int(hold_year), 2, 1), top.get_ylim()[1], "hold-out", color=HOLD, va="top", fontsize=9)
    return _finish(fig, "How to read: each point is one calendar month. A drifting level, a seasonal wave or a sudden jump "
                        "shows that the target is not the same kind of thing in every year.")


# --- 03 / 05 profile lines (hour of day, month) -----------------------------------

def plot_profile_line(profile, tick_labels, xlabel, title, caption, is_binary, label, threshold):
    """Top: share of late trains (95% band) over grey bars of train counts.
    Bottom (numeric target only): median, 75th and 90th percentile of the delay in minutes."""
    if profile is None or len(profile) < 2:
        return None
    x = profile["key"].to_numpy()
    if is_binary:
        fig, top = plt.subplots(figsize=(10, 5.2))
        bottom = None
    else:
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7.4), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    top.fill_between(x, profile["share_low"], profile["share_high"], color=DEV, alpha=0.2, label="95% band")
    top.plot(x, profile["share"], color=DEV, marker="o", linewidth=1.8, label="share of trains")
    top.set_ylabel("share of trains" if is_binary else f"share more than {threshold:g} min late")
    top.set_ylim(bottom=0)
    top.set_title(title)
    bars = top.twinx()
    bars.bar(x, profile["n"], color="lightgray", alpha=0.5, width=0.8)
    bars.set_ylabel("trains", color="gray")
    top.set_zorder(1)
    top.patch.set_visible(False)
    top.legend(loc="upper left", fontsize=8)
    axis_for_ticks = top if bottom is None else bottom
    if bottom is not None:
        bottom.plot(x, profile["center"], color=DEV, marker="o", markersize=3, linewidth=1.5, label="median")
        bottom.plot(x, profile["high"], color=DEV, linestyle=":", linewidth=1.3, label="75th percentile")
        bottom.plot(x, profile["p90"], color="tab:red", linestyle="--", linewidth=1.3, label="90th percentile")
        bottom.axhline(0, color="black", linewidth=0.6)
        bottom.set_ylabel(label)
        bottom.legend(loc="upper left", fontsize=8)
    axis_for_ticks.set_xticks(x)
    axis_for_ticks.set_xticklabels(tick_labels, fontsize=8)
    axis_for_ticks.set_xlabel(xlabel)
    return _finish(fig, caption)


# --- 04 weekday x hour heatmap ----------------------------------------------------

def plot_weekday_hour_heatmap(share, counts, min_count, threshold, is_binary):
    """share / counts: DataFrames, weekday 1..7 (rows) by hour 0..23 (columns)."""
    if share is None or share.empty:
        return None
    data = np.ma.masked_where((counts.to_numpy() < min_count) | np.isnan(share.to_numpy()), share.to_numpy())
    fig, ax = plt.subplots(figsize=(11, 4.6))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(range(share.shape[1]))
    ax.set_xticklabels([int(h) for h in share.columns], fontsize=8)
    ax.set_yticks(range(share.shape[0]))
    ax.set_yticklabels([WEEKDAY_NAMES[int(d) - 1] for d in share.index])
    ax.set_xlabel("hour of day")
    ax.set_title("When trains are late: weekday by hour")
    what = "share of trains" if is_binary else f"share more than {threshold:g} min late"
    fig.colorbar(im, ax=ax, label=what)
    return _finish(fig, f"How to read: each cell is the {what} among the trains scheduled in that weekday and hour. "
                        f"Blank cells have fewer than {min_count} trains.")


# --- 06 wind direction polar ------------------------------------------------------

def plot_wind_polar(profile, labels, calm, threshold, is_binary):
    """profile: per sector (key 0..n-1); calm: dict(n, share) or None."""
    if profile is None or profile.empty:
        return None
    n_sectors = len(labels)
    width = 2 * np.pi / n_sectors
    theta = profile["key"].to_numpy() * width
    what = "share of trains" if is_binary else f"share more than {threshold:g} min late"
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2), subplot_kw={"projection": "polar"})
    calm_note = (f"\n(calm, no direction: {int(calm['n']):,} trains, share {calm['share']:.2f})"
                 if calm and calm.get("n") else "")
    left_title = "how many trains (wind blowing from)" + (f"\n(calm, no direction: {int(calm['n']):,} trains)" if calm_note else "")
    for ax, values, title, color in ((axes[0], profile["n"], left_title, "tab:gray"),
                                     (axes[1], profile["share"], what + calm_note, DEV)):
        ax.bar(theta, values, width=width * 0.9, color=color, alpha=0.85)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(np.arange(n_sectors) * width)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_rlabel_position(157.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(title, fontsize=10, pad=22)
    fig.suptitle("Wind direction: how often, and how much delay", fontsize=12)
    return _finish(fig, "How to read: north is at the top and the compass is a circle, so 359 degrees and 1 degree are "
                        "neighbours. This is why the model gets the wind as sin and cos, not as a plain 0-360 number.", top=0.93)


# --- 07 delay vs each weather variable --------------------------------------------

def plot_binned_panels(panels, threshold, is_binary):
    """panels: list of {name, x, share, low, high, rho} (one row per bin of the variable)."""
    if not panels:
        return None
    fig, axes = _grid(len(panels), sharey=True)
    ymax = max(float(np.nanmax(p["high"])) for p in panels) * 1.05
    for ax, p in zip(axes, panels):
        order = np.argsort(p["x"])           # bins are coded heavy values first; draw left to right
        x = np.asarray(p["x"])[order]
        ax.fill_between(x, np.asarray(p["low"])[order], np.asarray(p["high"])[order], color=DEV, alpha=0.2)
        ax.plot(x, np.asarray(p["share"])[order], color=DEV, marker="o", markersize=3, linewidth=1.4)
        ax.set_title(f"{p['name']}\nSpearman rho = {p['rho']:+.2f}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_ylim(0, ymax)
    what = "share of trains" if is_binary else f"share more than {threshold:g} min late"
    for ax in axes[::4]:
        ax.set_ylabel(what, fontsize=8)
    fig.suptitle("How often trains are late across the range of each continuous variable", fontsize=11)
    return _finish(fig, "How to read: each panel cuts one variable into groups of trains (ten equal-sized groups, plus a group of its "
                        "own for any heavily repeated value such as 0 or -1) and shows how often trains are late in each (band = 95%). "
                        "A flat line means this variable alone says little. Panels are the variables with the largest |rho|.")


# --- 08 feature distributions -----------------------------------------------------

def plot_distribution_panels(items):
    """items: list of {name, dev, hold, notes} with raw values."""
    if not items:
        return None
    fig, axes = _grid(len(items))
    for ax, it in zip(axes, items):
        dev, hold = np.asarray(it["dev"], float), np.asarray(it["hold"], float)
        dev, hold = dev[~np.isnan(dev)], hold[~np.isnan(hold)]
        pooled = np.concatenate([dev, hold])
        if pooled.size == 0:
            continue
        lo, hi = np.percentile(pooled, [0.5, 99.5])
        bins = np.linspace(lo, hi, 41) if hi > lo else 20
        ax.hist(dev, bins=bins, density=True, histtype="stepfilled", alpha=0.45, color=DEV)
        if hold.size:
            ax.hist(hold, bins=bins, density=True, histtype="step", linewidth=1.5, color=HOLD)
        ax.set_title(it["name"], fontsize=8)
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
        if it.get("notes"):
            ax.text(0.97, 0.95, "\n".join(it["notes"]), transform=ax.transAxes, ha="right", va="top", fontsize=7,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "lightgray"})
    handles = [Patch(color=DEV, alpha=0.45, label="development years"), Patch(fill=False, edgecolor=HOLD, label="hold-out year")]
    fig.legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle("Distribution of each continuous variable, in its own units", fontsize=11)
    return _finish(fig, "How to read: shape of each variable (x axis trimmed to the 0.5-99.5 percentiles). Spikes are real: a 'zero' "
                        "or '-1' share is printed when it is large. Orange differing from blue means the new year looks different.")


# --- 09 Spearman heatmap, 10 dendrogram -------------------------------------------

def plot_spearman_heatmap(corr, annotate_max):
    n = len(corr)
    if n < 2:
        return None
    side = max(7.0, 0.42 * n + 2)
    fig, ax = plt.subplots(figsize=(side, side * 0.9))
    # the matrix is symmetric: show only the lower triangle and the diagonal
    lower = np.ma.masked_array(corr.to_numpy(), mask=np.triu(np.ones((n, n), dtype=bool), k=1))
    im = ax.imshow(lower, cmap="coolwarm", vmin=-1, vmax=1)
    for side_spine in ("top", "right"):
        ax.spines[side_spine].set_visible(False)
    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=max(5, 9 - n // 12))
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.index, fontsize=max(5, 9 - n // 12))
    if n <= annotate_max:
        for i in range(n):
            for j in range(i + 1):
                v = corr.iat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(v) > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Spearman rho", fraction=0.046)
    ax.set_title("Which variables move together (rank correlation)")
    return _finish(fig, "How to read: red = rise together, blue = move oppositely, white = unrelated. Variables are ordered so "
                        "that similar ones sit next to each other; only the lower triangle is drawn because the matrix is symmetric. Sin/cos columns of one pair are unrelated by construction.")


def plot_dendrogram(clustering, threshold):
    if clustering["linkage"] is None:
        return None
    labels = clustering["labels"]
    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.22)))
    dendrogram(clustering["linkage"], labels=labels, orientation="right", leaf_font_size=7,
               color_threshold=1.0 - threshold, ax=ax)
    ax.axvline(1.0 - threshold, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("1 - |Spearman rho|   (left of the red line = same cluster)")
    ax.set_title("Spearman clustering of the continuous variables")
    return _finish(fig, f"How to read: variables that join to the left of the red line (|rho| >= {threshold:g}) carry almost the "
                        "same ranking information; keeping one of them loses little. Pairs and one-hot families are not clustered.")


# --- 11 association with the target, 12 by year, 13 drift -------------------------

def plot_association(table, label):
    t = table.dropna(subset=["eta2"]).head(30)
    if t.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, max(4, 0.34 * len(t) + 1.5)))
    ax.barh(range(len(t)), t["eta2"], color=[KIND_COLORS[k] for k in t["kind"]])
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t["variable"], fontsize=8)
    ax.invert_yaxis()
    chance = float(np.nanmax(t["chance_level"]))
    ax.axvline(chance, color="gray", linestyle=":", linewidth=1)
    ax.text(chance, len(t) - 0.4, "  chance level", color="gray", fontsize=7, va="bottom")
    for i, v in enumerate(t["eta2"]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=7)
    ax.set_xlabel("share of the rank variance of the target explained by this variable alone")
    ax.set_title("How strongly each variable is tied to the target")
    ax.legend(handles=[Patch(color=c, label=k) for k, c in KIND_COLORS.items() if k in set(t["kind"])],
              fontsize=8, loc="lower right")
    return _finish(fig, "How to read: a long bar means that knowing this variable alone tells you a lot about where a train's delay "
                        "ranks. One variable at a time: two bars for related variables overlap, and a bar is not a causal effect.")


def plot_association_by_year(table, hold_year):
    if table is None or table.shape[1] < 2:
        return None
    order = table.mean(axis=1).sort_values(ascending=False).index[:30]
    t = table.loc[order]
    fig, ax = plt.subplots(figsize=(max(7, 0.8 * t.shape[1] + 4), max(4, 0.32 * len(t) + 1.5)))
    im = ax.imshow(t.to_numpy(), aspect="auto", cmap="magma")
    ax.set_xticks(range(t.shape[1]))
    ax.set_xticklabels([f"{c}\n(hold-out)" if hold_year is not None and c == hold_year else str(c) for c in t.columns], fontsize=8)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t.index, fontsize=8)
    if t.size <= 260:
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                v = t.iat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v < t.max().max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, label="share of rank variance explained")
    ax.set_title("Is what explains the target stable from year to year?")
    return _finish(fig, "How to read: same measure as the previous chart, computed separately for each year. A row that keeps its "
                        "colour is a stable pattern; a row that fades or brightens in one year is not. Blank = not computable "
                        "(the variable did not vary enough in that year).")


def plot_drift(table):
    if table is None or table.empty:
        return None
    t = table.head(30)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.34 * len(t) + 1.5)))
    ax.barh(range(len(t)), t["ks"], color=HOLD)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t["variable"], fontsize=8)
    ax.invert_yaxis()
    for i, row in enumerate(t.itertuples()):
        note = f" {row.ks:.3f}" + ("" if np.isnan(row.median_shift_iqr) else f"   (median {row.median_shift_iqr:+.2f} IQR)")
        ax.text(row.ks, i, note, va="center", fontsize=7)
    ax.set_xlim(0, max(0.1, float(t["ks"].max()) * 1.35))
    ax.set_xlabel("KS statistic: largest gap between the two cumulative distributions (0 = identical)")
    ax.set_title("Which variables look different in the hold-out year")
    return _finish(fig, "How to read: a long bar means the newest year's values are distributed differently from the earlier years "
                        "(for example a new sensor or a warmer winter). The IQR note says how far the typical value moved.")

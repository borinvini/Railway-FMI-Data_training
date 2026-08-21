# FCLL-Net feasibility measurements

Measured 2026-08-21 with `scripts/fcll_feasibility_check.py` on the full local
input set (96 months, `data/input/matched_data_flat_*.parquet`). The design doc
flagged two quantities as uncertain; both are now measured, and both come back
in favour of the design as specified.

## 1. The M proxy — distinct `commercialTrack` per station

| statistic | value |
|---|---|
| stations measured | 507 |
| mean distinct tracks | 5.34 |
| median | 4 |
| 25th / 75th percentile | 2 / 7 |
| max | 90 |
| **stations with >1 track** | **87.0%** |

**Decision rule:** drop `M` if fewer than 25% of stations have more than one
distinct track. 87.0% do, so **`M` is not dropped on these grounds.**

The one-month sample that produced "median 1, 75th percentile 2" was
misleading, exactly as suspected: a single month simply does not observe every
platform a station uses. Over 96 months the median station shows 4 distinct
tracks.

**Caveat — `M` is nonetheless absent from the current NF vector.** The
non-operational feature vector for `fcll_net` is defined by the scenario
catalogue in `config/scenarios.txt`, and no scenario lists a track count. So
although the measurement says `M` carries information, including it would mean
adding a feature to all 13 catalogues, which changes what the grid compares.
That is a decision for the author, not one this implementation made silently.

## 2. Sequence coverage

Over the first 12 months (10,521 target samples at `TARGET_STATION_CODE`):

| sequence | full-length share |
|---|---|
| Z (5 preceding stations) | **65.8%** |
| H (5 preceding trains) | **99.4%** |

**Decision rules and outcomes:**

- Full-Z below 50% → reduce `Z`. At 65.8%, **`Z = 5` stands.** The remaining
  34% are short-route and originating trains, which are pre-padded and masked
  rather than dropped, so they still train the station branch on whatever real
  history they have.
- Full-H below 50% → the target station's traffic is too sparse for the
  train-interaction branch. At 99.4%, **the H branch is well supported.** OL
  sees enough traffic that almost every arrival has five trains ahead of it.
  This is the branch carrying the paper's central claim, so this is the
  measurement that most needed to come back high.

## 3. Residual gap found while measuring — NF coverage per scenario

Not one of the two flagged risks, but discovered here and worth recording. The
raw event files do not carry the features preprocessing derives downstream, and
`fcll_net` skips those stages by design. `src/fcll_sequences.py` therefore
reproduces the calendar sin/cos features and the WMO 4680 `wawa_group` one-hots
directly from the raw timestamps and present-weather codes, importing the same
constants preprocessing uses so the two paths cannot drift.

After that, NF coverage per scenario is:

| scenario | NF columns available | note |
|---|---|---|
| 1 | 88 / 128 | missing the `weather_scenario_*` one-hots |
| 2–8 | full | |
| 9–12 | full | |
| 13 | full | |

**Still missing:** the `weather_scenario_*` one-hots used by scenario 1. These
come from a threshold classifier over weather values
(`add_weather_scenarios_col`) which is not reproduced here. Scenario 1's
`fcll_net` cell therefore runs with 88 of its 128 non-operational features. Every
other scenario is complete. Reproducing that classifier is the obvious next step
if scenario 1 turns out to matter for the comparison.

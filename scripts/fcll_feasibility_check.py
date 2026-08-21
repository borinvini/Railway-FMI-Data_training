"""Measure the two inputs the FCLL-Net design flagged as uncertain.

1. The M proxy (distinct commercialTrack per station). A one-month sample gave
   median 1, 75th percentile 2 — likely noise. Measured over all 96 months here.
2. Sequence coverage: what fraction of OL arrivals actually have five preceding
   stations and five preceding trains? Heavy padding weakens both LSTMs.

Run from the repo root: python scripts/fcll_feasibility_check.py
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.const_preprocessing import TARGET_STATION_CODE  # noqa: E402
from config.const_training import FCLL_H_TRAINS, FCLL_Z_STATIONS  # noqa: E402
from src.fcll_sequences import (assemble_sequences,  # noqa: E402
                                derive_station_visits)

files = sorted(glob.glob("data/input/matched_data_flat_*.parquet"))
if not files:
    sys.exit("no data/input/matched_data_flat_*.parquet found — run from the repo root")

tracks = {}
for path in files:
    df = pd.read_parquet(path, columns=["stationShortCode", "commercialTrack"])
    for station, group in df.dropna(subset=["commercialTrack"]).groupby("stationShortCode"):
        tracks.setdefault(station, set()).update(group["commercialTrack"].unique())
counts = pd.Series({k: len(v) for k, v in tracks.items()})
print(f"M proxy over all {len(files)} months:")
print(counts.describe().to_string())
print(f"stations with >1 track: {(counts > 1).mean():.1%}")

z_full = h_full = total = 0
sample = files[:12]
for path in sample:
    visits = derive_station_visits(pd.read_parquet(path))
    out = assemble_sequences(visits, target_station=TARGET_STATION_CODE, nf_columns=[])
    z_full += int((out["z_mask"].sum(axis=1) == FCLL_Z_STATIONS).sum())
    h_full += int((out["h_mask"].sum(axis=1) == FCLL_H_TRAINS).sum())
    total += len(out["label"])
print(f"\nsequence coverage over {len(sample)} months ({total:,} samples):")
print(f"  full Z ({FCLL_Z_STATIONS} stations): {z_full / total:.1%}")
print(f"  full H ({FCLL_H_TRAINS} trains):   {h_full / total:.1%}")

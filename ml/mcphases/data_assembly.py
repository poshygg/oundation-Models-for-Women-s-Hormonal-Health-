"""Assemble the mcPHASES daily feature table.

Base = hormones_and_selfreport.csv (one row per subject-day, carries the 4-phase
label + hormones + symptom diary). Wearable / CGM tables are aggregated to daily
statistics and left-joined on (id, study_interval, day_in_study).

Output: data/processed/mcphases_daily.parquet
"""
from __future__ import annotations
import pathlib
import pandas as pd
import numpy as np

RAW = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw" / "mcphases"
OUT = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

KEYS = ["id", "study_interval", "day_in_study"]
DROP = {"is_weekend", "timestamp"}


def _agg_daily(df: pd.DataFrame, keys: list[str], stats=("mean",)) -> pd.DataFrame:
    """Aggregate all numeric non-key columns to daily statistics."""
    num = [c for c in df.columns if c not in keys and c not in DROP
           and pd.api.types.is_numeric_dtype(df[c])]
    if not num:
        return pd.DataFrame(columns=keys)
    g = df.groupby(keys, as_index=False)[num].agg(list(stats))
    # flatten multiindex columns
    g.columns = keys + [f"{c}_{s}" if len(stats) > 1 else c
                        for c in num for s in stats]
    return g


def load_base() -> pd.DataFrame:
    h = pd.read_csv(RAW / "hormones_and_selfreport.csv")
    h = h.dropna(subset=["phase"]).copy()
    return h


def add_table(base, fname, rename=None, keys=KEYS, stats=("mean",), prefix=None):
    df = pd.read_csv(RAW / fname)
    if rename:
        df = df.rename(columns=rename)
    if not set(keys).issubset(df.columns):
        print(f"  skip {fname}: missing keys {set(keys)-set(df.columns)}")
        return base
    agg = _agg_daily(df, keys, stats)
    feat_cols = [c for c in agg.columns if c not in keys]
    if prefix:
        agg = agg.rename(columns={c: f"{prefix}_{c}" for c in feat_cols})
    merged = base.merge(agg, on=keys, how="left")
    print(f"  + {fname:38} -> +{len(feat_cols)} cols")
    return merged


def main():
    base = load_base()
    print(f"base (hormones+selfreport, labeled): {base.shape}")

    # daily-resolution tables (already one row per day)
    base = add_table(base, "resting_heart_rate.csv", prefix="rhr")
    base = add_table(base, "sleep_score.csv", prefix="sleep")
    base = add_table(base, "stress_score.csv", prefix="stress")
    base = add_table(base, "respiratory_rate_summary.csv", prefix="resp")
    base = add_table(base, "time_in_heart_rate_zones.csv", prefix="hrzone")
    base = add_table(base, "active_minutes.csv", prefix="act")
    base = add_table(base, "demographic_vo2_max.csv", prefix="vo2")

    # nightly skin temperature: keyed on sleep_start_day_in_study
    base = add_table(base, "computed_temperature.csv",
                     rename={"sleep_start_day_in_study": "day_in_study"},
                     prefix="temp")
    # wrist temperature diff from baseline (timestamped -> daily mean)
    base = add_table(base, "wrist_temperature.csv", prefix="wtemp")

    # high-frequency timestamped signals -> daily mean+std
    base = add_table(base, "heart_rate.csv", stats=("mean", "std"), prefix="hr")
    base = add_table(base, "heart_rate_variability_details.csv",
                     stats=("mean", "std"), prefix="hrv")
    base = add_table(base, "glucose.csv", stats=("mean", "std"), prefix="cgm")
    base = add_table(base, "steps.csv", stats=("sum",), prefix="steps")

    # tidy label
    base["phase"] = base["phase"].str.strip()
    print(f"\nassembled: {base.shape}")
    print("phase distribution:\n", base["phase"].value_counts())
    print("subjects:", base["id"].nunique(),
          "| interval split:", base["study_interval"].value_counts().to_dict())

    out = OUT / "mcphases_daily.parquet"
    base.to_parquet(out, index=False)
    base.to_csv(OUT / "mcphases_daily.csv", index=False)
    print(f"\nsaved -> {out}  ({base.shape[1]} cols)")


if __name__ == "__main__":
    main()

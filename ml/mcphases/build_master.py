"""Build the data-aligned master table for Challenge-05 phase classification.

Aligns with Specht et al. (mcPHASES SOTA) so our numbers are comparable, but keeps
flags instead of hard-dropping rows — because their symptom-ONLY setup forced them to
discard symptom-missing days, whereas our MULTIMODAL setup can still use those days
(wearables are present). One table, two views:

  * paper-replication view  : 2022 only, 14-day warmup dropped, rows inside a >=5-day
    all-symptom-missing run dropped, symptoms present  -> reproduce ~0.662 baseline.
  * multimodal view         : 2022 + warmup dropped, keep wearable-rich days even when
    symptoms are missing     -> our contribution (extra data + wearable channels).

Adds flags (is_warmup, sym_missing, in_missing_run5) and a leakage-free
days_since_bleed (derived from reported flow_volume, NOT the phase label).

Output: data/processed/mcphases_master_2022.parquet
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
SYMPTOMS = ["flow_volume", "flow_color", "appetite", "exerciselevel", "headaches",
            "cramps", "sorebreasts", "fatigue", "sleepissue", "moodswing", "stress",
            "foodcravings", "indigestion", "bloating"]
WARMUP_DAYS = 14
MISSING_RUN = 5


def _max_run_flags(missing: np.ndarray) -> np.ndarray:
    """Mark rows that sit inside a run of >= MISSING_RUN consecutive missing days."""
    in_run = np.zeros(len(missing), dtype=bool)
    i = 0
    while i < len(missing):
        if missing[i]:
            j = i
            while j < len(missing) and missing[j]:
                j += 1
            if j - i >= MISSING_RUN:
                in_run[i:j] = True
            i = j
        else:
            i += 1
    return in_run


def _days_since_flow(flow: pd.Series) -> list:
    """Leakage-free cycle position: days since last reported menstrual flow."""
    out, c = [], None
    for v in flow:
        bleed = pd.notna(v) and str(v).strip().lower() != "not at all"
        if bleed:
            c = 0
        elif c is not None:
            c += 1
        out.append(c if c is not None else 0.0)
    return out


def main():
    df = pd.read_parquet(PROC / "mcphases_daily.parquet")
    df = df[df["study_interval"] == 2022].copy()
    df = df.sort_values(["id", "day_in_study"]).reset_index(drop=True)

    df["sym_missing"] = df[SYMPTOMS].isna().all(axis=1)
    df["is_warmup"] = df["day_in_study"] <= WARMUP_DAYS

    parts = []
    for _, g in df.groupby("id", sort=False):
        g = g.sort_values("day_in_study").copy()
        g["in_missing_run5"] = _max_run_flags(g["sym_missing"].to_numpy())
        g["days_since_flow"] = _days_since_flow(g["flow_volume"])
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)

    paper = out[~out.is_warmup & ~out.in_missing_run5 & ~out.sym_missing]
    multi = out[~out.is_warmup]

    print(f"master (2022): {len(out)} rows, {out.id.nunique()} subjects")
    print(f"  paper-replication view : {len(paper)} rows, {paper.id.nunique()} subjects  "
          f"(target ~2983 / 41)")
    print(f"  multimodal view        : {len(multi)} rows, {multi.id.nunique()} subjects  "
          f"(+{len(multi) - len(paper)} rows kept via wearables)")
    print(f"  phase dist (multimodal): {multi.phase.value_counts().to_dict()}")

    dst = PROC / "mcphases_master_2022.parquet"
    out.to_parquet(dst, index=False)
    print(f"saved -> {dst}  ({out.shape[1]} cols; flags: is_warmup, sym_missing, "
          f"in_missing_run5, days_since_flow)")


if __name__ == "__main__":
    main()

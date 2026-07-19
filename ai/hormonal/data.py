"""ai/hormonal/data.py — mcPHASES loading + physiology-grounded features.

Challenge 05, Layer 02: 4-phase menstrual cycle classification + hormone regression.

Schema note: the canonical column names here ARE the real mcPHASES daily-table
names (`id`, `day_in_study`, `phase` in {Menstrual, Follicular, Fertility, Luteal},
`rhr_value`, `estrogen`, ...). Everything downstream — train.py, compare.py,
regression.py, explain.py — speaks this one schema. The synthetic fallback
generator emits the same names, so running with no data stays schema-consistent.

Two data paths, one interface:
  * If the processed mcPHASES daily table exists, load it.
  * Otherwise synthesize a physiology-grounded dataset so the whole pipeline runs
    end-to-end today.

Feature engineering (rolling stats, deltas, per-subject baseline deviation) is
computed WITHIN each recording segment (subject x study interval), and LOSO groups
by subject id, so nothing crosses the leave-one-subject-out boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

## Cycle order matches the B1 OOF probability columns (Menstrual=0 ... Luteal=3).
PHASES = ["Menstrual", "Follicular", "Fertility", "Luteal"]
PHASE_TO_IDX = {p: i for i, p in enumerate(PHASES)}
PHASE_DURATIONS = {"Menstrual": 5, "Follicular": 8, "Fertility": 2, "Luteal": 13}

## Canonical schema keys (real mcPHASES daily-table names).
SUBJECT_COL = "id"
TIME_COL = "day_in_study"
LABEL_COL = "phase"
INTERVAL_COL = "study_interval"          # a subject may have two recording segments
REGULARITY_COL = "cycle_regular"         # not in real data -> defaults to True

## Regression targets: Mira urinary metabolites. PdG is sparse (dropped in some
## intervals); regression masks per-hormone NaNs.
HORMONE_COLS = ["lh", "estrogen", "pdg"]

## Default feature columns from the real daily table, chosen for coverage +
## physiological relevance. engineer() uses the intersection with what's present,
## so the synthetic subset and the full real table both work.
NUMERIC_COLS = [
    "rhr_value", "sleep_resting_heart_rate", "hr_bpm_mean", "hr_bpm_std",
    "hrv_rmssd_mean", "hrv_high_frequency_mean", "hrv_low_frequency_mean",
    "resp_full_sleep_breathing_rate", "wtemp_temperature_diff_from_baseline",
    "temp_nightly_temperature", "sleep_overall_score", "sleep_deep_sleep_in_minutes",
    "sleep_restlessness", "stress_stress_score", "steps_steps", "act_very",
    "act_moderately", "vo2_filtered_demographic_vo2_max",
    "cgm_glucose_value_mean", "cgm_glucose_value_std",
]
## Self-report symptoms — ordinal severity strings in real data, encoded below.
ORDINAL_COLS = [
    "flow_volume", "cramps", "bloating", "moodswing", "sorebreasts", "fatigue",
    "headaches", "foodcravings", "indigestion", "sleepissue", "appetite", "exerciselevel",
    "stress",
]

## Ordinal severity encoding shared by every self-report column (lowercased keys).
SEVERITY_MAP = {
    "not at all": 0.0,
    "spotting / very light": 1.0, "very low/little": 1.0, "very low": 1.0,
    "somewhat light": 2.0, "light": 2.0, "low": 2.0,
    "moderate": 3.0,
    "somewhat heavy": 4.0, "high": 4.0,
    "heavy": 5.0, "very high": 5.0,
    "very heavy": 6.0,
}


@dataclass
class Dataset:
    X: pd.DataFrame          # engineered features, one row per subject-day
    y: np.ndarray            # int phase labels, aligned to X rows
    groups: np.ndarray       # subject id per row (LOSO grouping key)
    regular: np.ndarray      # bool per row: is this subject's cycle regular
    feature_names: list[str]
    synthetic: bool
    y_reg: np.ndarray | None = None      # (n, 3) hormone levels [lh, estrogen, pdg], if available
    hormone_names: list[str] | None = None
    segment: np.ndarray | None = None    # (n,) contiguous (subject, interval) id; rows are day-ordered within it


## ----------------------------------------------------------------------------
## Loading
## ----------------------------------------------------------------------------
def load_raw(cfg: dict) -> tuple[pd.DataFrame, bool]:
    """Return (long raw table, is_synthetic) in the canonical schema."""
    d = cfg["data"]
    path = Path(d["processed_table"])
    if path.exists():
        raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        return _standardize_columns(raw, d), False
    raw = _synthesize(
        n_subjects=int(d.get("synth_subjects", 41)),
        days=int(d.get("synth_days_per_subject", 90)),
        irregular_frac=float(d.get("synth_irregular_frac", 0.25)),
        seed=int(cfg["experiment"]["seed"]),
    )
    return raw, True


def _standardize_columns(raw: pd.DataFrame, d: dict) -> pd.DataFrame:
    """Map the table's columns to the canonical schema. Real mcPHASES already uses
    these names; the config may override subject/time/label/interval keys."""
    ren = {}
    for canon, key in ((SUBJECT_COL, "subject_col"), (TIME_COL, "time_col"),
                       (LABEL_COL, "label_col"), (INTERVAL_COL, "interval_col")):
        src = d.get(key)
        if src and src in raw.columns and src != canon:
            ren[src] = canon
    raw = raw.rename(columns=ren)

    if INTERVAL_COL not in raw.columns:
        raw[INTERVAL_COL] = 0
    if REGULARITY_COL not in raw.columns:
        ## Real data has no regularity flag; treat all as regular (by-regularity
        ## reporting then collapses to one group).
        raw[REGULARITY_COL] = True
    if raw[LABEL_COL].dtype == object:
        raw["phase_idx"] = raw[LABEL_COL].map(PHASE_TO_IDX)
    else:
        raw["phase_idx"] = raw[LABEL_COL].astype(int)
    ## Drop rows whose phase is not one of the four canonical labels.
    return raw[raw["phase_idx"].notna()].reset_index(drop=True).assign(
        phase_idx=lambda x: x["phase_idx"].astype(int))


## ----------------------------------------------------------------------------
## Synthetic generator (physiology-grounded, real column names)
## ----------------------------------------------------------------------------
def _phase_sequence(days: int, rng: np.random.Generator, regular: bool) -> np.ndarray:
    seq: list[int] = []
    while len(seq) < days:
        durs = dict(PHASE_DURATIONS)
        if not regular:
            durs["Follicular"] = max(2, int(rng.normal(8, 4)))
            durs["Luteal"] = max(6, int(rng.normal(13, 4)))
            durs["Menstrual"] = max(2, int(rng.normal(5, 1.5)))
            if rng.random() < 0.25:
                durs["Fertility"] = 0  # anovulatory cycle
        for phase in PHASES:
            seq.extend([PHASE_TO_IDX[phase]] * durs[phase])
    return np.array(seq[:days])


## Per-phase mean offsets on top of each subject's baseline (real column names).
_PHASE_EFFECTS = {
    "wtemp_temperature_diff_from_baseline": {"Menstrual": -0.05, "Follicular": -0.10, "Fertility": -0.15, "Luteal": 0.35},
    "rhr_value":       {"Menstrual": 0.5,  "Follicular": -1.0, "Fertility": 0.0,  "Luteal": 2.5},
    "hr_bpm_mean":     {"Menstrual": 0.5,  "Follicular": -0.8, "Fertility": 0.0,  "Luteal": 2.0},
    "hrv_rmssd_mean":  {"Menstrual": -1.0, "Follicular": 3.0,  "Fertility": 1.0,  "Luteal": -5.0},
    "resp_full_sleep_breathing_rate": {"Menstrual": 0.1, "Follicular": -0.2, "Fertility": 0.0, "Luteal": 0.6},
    "sleep_overall_score": {"Menstrual": -3.0, "Follicular": 1.0, "Fertility": 0.0, "Luteal": -2.0},
    "stress_stress_score": {"Menstrual": 6.0, "Follicular": -3.0, "Fertility": 0.0, "Luteal": 5.0},
    "steps_steps":     {"Menstrual": -800, "Follicular": 300,  "Fertility": 200,  "Luteal": -300},
    "cgm_glucose_value_mean": {"Menstrual": 1.0, "Follicular": -2.0, "Fertility": 0.0, "Luteal": 3.0},
    "cgm_glucose_value_std":  {"Menstrual": 0.5, "Follicular": 0.0, "Fertility": 0.0, "Luteal": 1.0},
    "flow_volume":     {"Menstrual": 4.0,  "Follicular": 0.0,  "Fertility": 0.0,  "Luteal": 0.0},
    "cramps":          {"Menstrual": 3.0,  "Follicular": 0.0,  "Fertility": 0.3,  "Luteal": 0.8},
    "bloating":        {"Menstrual": 1.5,  "Follicular": 0.0,  "Fertility": 0.2,  "Luteal": 2.0},
    "moodswing":       {"Menstrual": 1.0,  "Follicular": -0.5, "Fertility": 0.0,  "Luteal": 2.0},
    "sorebreasts":     {"Menstrual": 0.5,  "Follicular": 0.0,  "Fertility": 0.3,  "Luteal": 2.2},
    "lh":              {"Menstrual": -12.0, "Follicular": -5.0,  "Fertility": 55.0, "Luteal": -8.0},
    "estrogen":        {"Menstrual": -60.0, "Follicular": 20.0,  "Fertility": 110.0, "Luteal": 30.0},
    "pdg":             {"Menstrual": -15.0, "Follicular": -12.0, "Fertility": -3.0,  "Luteal": 45.0},
}
_BASELINE = {
    "wtemp_temperature_diff_from_baseline": (0.0, 0.15),
    "rhr_value": (62.0, 2.5), "hr_bpm_mean": (70.0, 3.0), "hrv_rmssd_mean": (45.0, 6.0),
    "resp_full_sleep_breathing_rate": (15.5, 0.5), "sleep_overall_score": (80.0, 5.0),
    "stress_stress_score": (70.0, 10.0), "steps_steps": (7500.0, 1500.0),
    "cgm_glucose_value_mean": (98.0, 4.0), "cgm_glucose_value_std": (18.0, 3.0),
    "flow_volume": (0.0, 0.3), "cramps": (0.3, 0.6), "bloating": (0.4, 0.6),
    "moodswing": (1.5, 0.8), "sorebreasts": (0.3, 0.5),
    "lh": (25.0, 5.0), "estrogen": (120.0, 10.0), "pdg": (25.0, 5.0),
}
_SUBJECT_SD = {
    "wtemp_temperature_diff_from_baseline": 0.25, "rhr_value": 6.0, "hr_bpm_mean": 6.0,
    "hrv_rmssd_mean": 10.0, "resp_full_sleep_breathing_rate": 0.8, "sleep_overall_score": 6.0,
    "stress_stress_score": 10.0, "steps_steps": 1500.0, "cgm_glucose_value_mean": 8.0,
    "cgm_glucose_value_std": 3.0, "flow_volume": 0.3, "cramps": 0.4, "bloating": 0.4,
    "moodswing": 0.8, "sorebreasts": 0.4, "lh": 6.0, "estrogen": 15.0, "pdg": 6.0,
}
SYNTH_COLS = list(_BASELINE.keys())
_NONNEG = {"flow_volume", "cramps", "bloating", "moodswing", "sorebreasts",
           "steps_steps", "stress_stress_score", "lh", "estrogen", "pdg"}


def _synthesize(n_subjects: int, days: int, irregular_frac: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    n_irregular = int(round(n_subjects * irregular_frac))
    irregular_ids = set(rng.choice(n_subjects, size=n_irregular, replace=False).tolist())

    for sid in range(n_subjects):
        regular = sid not in irregular_ids
        subj_offset = {c: rng.normal(0, _SUBJECT_SD[c]) for c in SYNTH_COLS}
        noise_mult = 1.0 if regular else 1.8
        phase_seq = _phase_sequence(days, rng, regular)

        for day, pidx in enumerate(phase_seq):
            phase = PHASES[pidx]
            row = {SUBJECT_COL: sid, INTERVAL_COL: 0, TIME_COL: day, LABEL_COL: phase,
                   "phase_idx": pidx, REGULARITY_COL: regular}
            for c in SYNTH_COLS:
                base, noise = _BASELINE[c]
                val = base + subj_offset[c] + _PHASE_EFFECTS[c][phase] + rng.normal(0, noise * noise_mult)
                if c in _NONNEG:
                    val = max(0.0, val)
                if c == "sleep_overall_score":
                    val = float(np.clip(val, 0.0, 100.0))
                row[c] = val
            rows.append(row)

    return pd.DataFrame(rows)


## ----------------------------------------------------------------------------
## Feature engineering (within-segment; LOSO-safe)
## ----------------------------------------------------------------------------
def _encode_ordinal(series: pd.Series) -> pd.Series:
    """Ordinal severity -> float. Numeric input (synthetic) passes through;
    string input (real self-report) maps via SEVERITY_MAP; unknowns -> NaN."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    mapped = series.astype(str).str.strip().str.lower().map(SEVERITY_MAP)
    return mapped.astype(float)


def _select_cols(cfg: dict, raw: pd.DataFrame) -> list[str]:
    fs = cfg["features"]["feature_set"]
    numeric = [c for c in NUMERIC_COLS if c in raw.columns]
    ordinal = [c for c in ORDINAL_COLS if c in raw.columns]
    if fs == "symptoms_only":
        return ordinal
    if fs == "wearables_only":
        return numeric
    return numeric + ordinal


def engineer(raw: pd.DataFrame, cfg: dict) -> Dataset:
    fcfg = cfg["features"]
    cols = _select_cols(cfg, raw)
    windows = list(fcfg["rolling_windows"])
    raw = raw.sort_values([SUBJECT_COL, INTERVAL_COL, TIME_COL]).reset_index(drop=True)

    ## Base signals as floats (ordinal encoded). NaN is PRESERVED, not imputed:
    ## every backend we use (LightGBM / XGBoost / CatBoost / EBM / TabPFN) handles
    ## missing natively and can split on it, so "not reported" (a ~42% signal for
    ## self-report symptoms) and "device absent" (CGM in interval 2) stay informative
    ## instead of being papered over with a fabricated median. Rolling stats below
    ## use min_periods=1 so they still produce a value wherever the window has data.
    seg = [SUBJECT_COL, INTERVAL_COL]
    base = pd.DataFrame({SUBJECT_COL: raw[SUBJECT_COL].values, INTERVAL_COL: raw[INTERVAL_COL].values})
    for c in cols:
        base[c] = _encode_ordinal(raw[c]) if c in ORDINAL_COLS else raw[c].astype(float)

    ## Accumulate every derived column in a dict, then build the frame once
    ## (avoids the fragmentation from hundreds of individual inserts).
    out: dict[str, np.ndarray] = {c: base[c].to_numpy() for c in cols}
    g = base.groupby(seg, sort=False)
    for c in cols:
        s = g[c]
        for w in windows:
            out[f"{c}_roll{w}_mean"] = s.transform(lambda x, w=w: x.rolling(w, min_periods=1).mean()).to_numpy()
            out[f"{c}_roll{w}_std"] = s.transform(
                lambda x, w=w: x.rolling(w, min_periods=1).std().fillna(0.0)).to_numpy()
        out[f"{c}_delta1"] = s.transform(lambda x: x.diff().fillna(0.0)).to_numpy()
    if fcfg.get("add_subject_baseline_deltas", True):
        for c in cols:
            b = g[c].transform(lambda x: x.quantile(0.25)).to_numpy()
            out[f"{c}_dev_base"] = base[c].to_numpy() - b

    ## Within-segment z-score: each person's signal RELATIVE to their own
    ## distribution. Measured +0.018 macro-F1 in the symptoms-only setting, but
    ## NEUTRAL for the full multimodal pipeline (wearables + dev_base already carry
    ## person-relative info), so it is opt-in rather than default.
    if fcfg.get("add_zscore", False):
        for c in cols:
            mu = g[c].transform("mean").to_numpy()
            sd = g[c].transform("std").to_numpy()
            sd = np.where(np.isnan(sd) | (sd == 0), np.nan, sd)   # NaN -> feature NaN (backends handle)
            out[f"{c}_z"] = (base[c].to_numpy() - mu) / sd

    out["days_since_bleed"] = _days_since_bleed(raw)
    if "is_weekend" in raw.columns:
        out["is_weekend"] = raw["is_weekend"].astype(float).to_numpy()
    else:
        out["is_weekend"] = (raw[TIME_COL].to_numpy() % 7 >= 5).astype(float)

    ## NaN preserved (see base-signal note above). is_weekend is always defined;
    ## days_since_bleed is NaN until the first detected period onset in a segment.
    feats = pd.DataFrame(out)
    feature_names = list(feats.columns)

    y_reg, hormone_names = None, None
    present_h = [h for h in HORMONE_COLS if h in raw.columns]
    if present_h:
        y_reg = raw.reindex(columns=HORMONE_COLS).to_numpy(dtype=float)  # NaN kept; regression masks it
        hormone_names = list(HORMONE_COLS)

    ## Contiguous segment id per row (rows already sorted by subject/interval/day),
    ## so temporal smoothing can decode each recording segment in day order.
    segment = pd.factorize(pd.MultiIndex.from_arrays(
        [raw[SUBJECT_COL].to_numpy(), raw[INTERVAL_COL].to_numpy()]))[0]

    return Dataset(
        X=feats.reset_index(drop=True),
        y=raw["phase_idx"].to_numpy(),
        groups=raw[SUBJECT_COL].to_numpy(),
        regular=raw[REGULARITY_COL].to_numpy().astype(bool),
        feature_names=feature_names,
        synthetic=False,
        y_reg=y_reg,
        hormone_names=hormone_names,
        segment=segment,
    )


## Cycle-day reconstruction knobs. A bleeding EPISODE onset requires flow at least
## this heavy, following at least this many non-bleeding days — so mid-cycle
## spotting and multi-day periods don't reset the clock. Tuned on the calendar-only
## LOSO diagnostic (0.425 -> 0.526 acc vs the naive any-flow reset).
_BLEED_MIN_SEVERITY = 2.0   # >= "Light" on the 0-6 SEVERITY_MAP (excludes spotting)
_BLEED_MIN_GAP = 10         # days; two onsets can't be closer than this


def _days_since_bleed(raw: pd.DataFrame) -> np.ndarray:
    """Days since the most recent menstrual-flow EPISODE onset, within each segment.

    Cycle position derived from the self-reported `flow_volume` symptom ONLY — never
    from `phase`/`phase_idx` (that would leak the target). LEAKAGE HISTORY: an earlier
    version counted days since the Menstrual *label*, which inflated LOSO macro-F1 by
    ~0.15 (0.52 honest -> 0.67 leaked); removed.

    Episode logic (not a naive any-flow reset): an onset is the first day of a
    bleeding run (flow >= Light) that follows >= _BLEED_MIN_GAP non-bleeding days, so
    spotting and long periods don't cause spurious resets. NaN before the first
    detected onset (cycle position unknown) — every backend handles the NaN.
    """
    if "flow_volume" not in raw.columns:
        return np.full(len(raw), np.nan)
    flow = _encode_ordinal(raw["flow_volume"]).to_numpy()
    out = np.full(len(raw), np.nan)
    for _, idx in raw.groupby([SUBJECT_COL, INTERVAL_COL], sort=False).groups.items():
        rows = list(idx)
        onset, last_bleed = None, -(10 ** 9)
        for i, r in enumerate(rows):
            bleed = (not np.isnan(flow[r])) and flow[r] >= _BLEED_MIN_SEVERITY
            if bleed and (i - last_bleed) >= _BLEED_MIN_GAP:
                onset = i
            if bleed:
                last_bleed = i
            out[r] = (i - onset) if onset is not None else np.nan
    return out


def build_dataset(cfg: dict) -> Dataset:
    raw, synthetic = load_raw(cfg)
    ds = engineer(raw, cfg)
    ds.synthetic = synthetic
    return ds

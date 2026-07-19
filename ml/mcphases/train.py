"""Train the mcPHASES 4-phase classifier under leave-one-subject-out (LOSO).

Baselines:
  B0  symptom-only  (reproduce the published CatBoost SOTA, ~67.6% / 0.662)
  B1  multimodal    (B0 + Fitbit temp/HR/HRV/sleep/stress/resp + CGM + per-subject z-scores)

Hormones (lh/estrogen/pdg) are NEVER used as features: the phase label is hormone-derived,
so using them would leak the ground truth.

Usage:  python ml/mcphases/train.py --baseline B0
        python ml/mcphases/train.py --baseline B1
Outputs out-of-fold predictions + per-day class probabilities to data/processed/oof_<baseline>.parquet
"""
from __future__ import annotations
import argparse, pathlib, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, classification_report, confusion_matrix)
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

KEYS = ["id", "study_interval", "day_in_study"]
LABEL = "phase"
HORMONES = ["lh", "estrogen", "pdg"]            # excluded from features (label leakage)
SYMPTOMS = ["flow_volume", "flow_color", "appetite", "exerciselevel", "headaches",
            "cramps", "sorebreasts", "fatigue", "sleepissue", "moodswing", "stress",
            "foodcravings", "indigestion", "bloating"]
CLASSES = ["Menstrual", "Follicular", "Fertility", "Luteal"]  # cyclic order

# Ordinal encodings (Likert -> integer) so CatBoost runs in fast numeric mode instead
# of the slow categorical target-statistics path.
GEN = {"Not at all": 0, "Very Low": 1, "Very Low/Little": 1, "Low": 2, "Moderate": 3,
       "High": 4, "Very High": 5, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
FV = {"Not at all": 0, "Spotting / Very Light": 1, "Light": 2, "Somewhat Light": 3,
      "Moderate": 4, "Somewhat Heavy": 5, "Heavy": 6, "Very Heavy": 7}
FC = {"Not at all": 0, "Bright Red": 1, "Dark Brown / Dark Red": 2, "Pink": 3, "Black": 4,
      "Grey": 5, "Orange": 6, "Yellow": 7, "Other": 8}


def encode_symptoms(df: pd.DataFrame) -> None:
    """In-place ordinal encoding of the symptom diary columns to numeric."""
    for c in SYMPTOMS:
        if c not in df.columns:
            continue
        s = df[c].astype("object")
        if c == "flow_volume":
            df[c] = s.map(FV)
        elif c == "flow_color":
            df[c] = s.map(FC)
        else:
            df[c] = s.map(GEN)
        df[c] = pd.to_numeric(df[c], errors="coerce")


TEMPORAL_SYMPTOMS = ["flow_volume", "cramps", "fatigue", "bloating", "moodswing",
                     "sorebreasts", "headaches", "appetite"]
TEMPORAL_PHYSIO = ["rhr_value", "temp_nightly_temperature",
                   "wtemp_temperature_diff_from_baseline", "resp_full_sleep_breathing_rate",
                   "hrv_rmssd_mean", "hr_bpm_mean", "sleep_resting_heart_rate",
                   "stress_stress_score"]


def _days_since_flow(series: pd.Series) -> list:
    """Days since the last reported menstrual flow (flow_volume ordinal > 0).
    Derived from the SYMPTOM diary, not the phase label -> no leakage."""
    out, c = [], None
    for v in series:
        if pd.notna(v) and v > 0:
            c = 0
        elif c is not None:
            c = c + 1
        out.append(c if c is not None else np.nan)
    return out


def add_temporal_features(df: pd.DataFrame, base_cols: list[str]) -> list[str]:
    """Per-subject/interval rolling (3,7-day) means, day-over-day delta, and a
    symptom-derived cycle-day proxy. Ordered by day_in_study within each series."""
    df.sort_values(["id", "study_interval", "day_in_study"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    g = df.groupby(["id", "study_interval"], sort=False)
    new: list[str] = []
    for c in base_cols:
        if c not in df.columns:
            continue
        df[f"{c}_r3"] = g[c].transform(lambda s: s.rolling(3, min_periods=1).mean())
        df[f"{c}_r7"] = g[c].transform(lambda s: s.rolling(7, min_periods=1).mean())
        df[f"{c}_d1"] = g[c].transform(lambda s: s.diff())
        new += [f"{c}_r3", f"{c}_r7", f"{c}_d1"]
    # cycle-day proxy from menstrual flow
    df["days_since_flow"] = g["flow_volume"].transform(
        lambda s: pd.Series(_days_since_flow(s), index=s.index))
    new.append("days_since_flow")
    return new


def add_subject_zscores(df: pd.DataFrame, num_cols: list[str]) -> list[str]:
    """Per-subject z-scores (computed on a subject's own history -> LOSO-safe)."""
    zcols = []
    g = df.groupby("id")
    for c in num_cols:
        m = g[c].transform("mean")
        s = g[c].transform("std").replace(0, np.nan)
        z = f"{c}__z"
        df[z] = (df[c] - m) / s
        zcols.append(z)
    return zcols


def build_features(df: pd.DataFrame, baseline: str):
    df = df.copy()
    encode_symptoms(df)                       # ordinal -> numeric (fast path)
    symptom_feats = [c for c in SYMPTOMS if c in df.columns]
    cat_features: list[str] = []              # everything numeric now

    if baseline == "B0":
        tcols = add_temporal_features(df, TEMPORAL_SYMPTOMS)
        feats = list(symptom_feats) + tcols
    elif baseline == "B1":
        wearable = [c for c in df.columns
                    if c not in KEYS + [LABEL] + HORMONES + SYMPTOMS + ["is_weekend"]
                    and pd.api.types.is_numeric_dtype(df[c])]
        tcols = add_temporal_features(df, TEMPORAL_SYMPTOMS + TEMPORAL_PHYSIO)
        zcols = add_subject_zscores(df, wearable)
        feats = list(symptom_feats) + wearable + zcols + tcols
    else:
        raise ValueError(baseline)
    return df, feats, cat_features


def run(baseline: str):
    df = pd.read_parquet(PROC / "mcphases_daily.parquet")
    df = df[df[LABEL].isin(CLASSES)].reset_index(drop=True)
    df, feats, cat_features = build_features(df, baseline)

    X, y, groups = df[feats], df[LABEL], df["id"]
    print(f"[{baseline}] {X.shape[0]} rows, {len(feats)} features "
          f"({len(cat_features)} categorical), {groups.nunique()} subjects")

    logo = LeaveOneGroupOut()
    oof_pred = np.empty(len(df), dtype=object)
    oof_prob = np.zeros((len(df), len(CLASSES)))
    cat_idx = [feats.index(c) for c in cat_features]

    for i, (tr, te) in enumerate(logo.split(X, y, groups)):
        model = CatBoostClassifier(
            iterations=400, depth=5, learning_rate=0.05,
            loss_function="MultiClass", auto_class_weights="Balanced",
            random_seed=42, verbose=False)
        model.fit(Pool(X.iloc[tr], y.iloc[tr], cat_features=cat_idx))
        p = model.predict_proba(Pool(X.iloc[te], cat_features=cat_idx))
        order = list(model.classes_)
        # reorder columns to CLASSES
        reorder = [order.index(c) for c in CLASSES]
        oof_prob[te] = p[:, reorder]
        oof_pred[te] = np.array(CLASSES)[oof_prob[te].argmax(1)]
        print(f"  fold {i+1:2d}/42  subject={groups.iloc[te].iloc[0]}  n={len(te)}", end="\r")

    acc = accuracy_score(y, oof_pred)
    bacc = balanced_accuracy_score(y, oof_pred)
    mf1 = f1_score(y, oof_pred, average="macro", labels=CLASSES)
    print("\n" + "=" * 60)
    print(f"[{baseline}]  accuracy={acc:.3f}  balanced_acc={bacc:.3f}  macro_F1={mf1:.3f}")
    print("(published SOTA reference: acc 0.676 / macro_F1 0.662)")
    print("-" * 60)
    print(classification_report(y, oof_pred, labels=CLASSES, digits=3, zero_division=0))
    print("confusion (rows=true, cols=pred; order=%s):" % CLASSES)
    print(confusion_matrix(y, oof_pred, labels=CLASSES))

    out = df[KEYS + [LABEL]].copy()
    out["pred"] = oof_pred
    for j, c in enumerate(CLASSES):
        out[f"prob_{c}"] = oof_prob[:, j]
    dst = PROC / f"oof_{baseline}.parquet"
    out.to_parquet(dst, index=False)
    print(f"\nsaved OOF -> {dst}")
    return acc, bacc, mf1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="B0", choices=["B0", "B1"])
    run(ap.parse_args().baseline)

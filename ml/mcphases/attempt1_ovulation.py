"""Attempt 1 — Fertility ovulation anchor.

The SOTA paper's worst class is Fertility (F1 0.462): the ovulation window is short
(~3 days) and its symptoms are subtle. Its Future Work names the fix: post-ovulatory
skin-temperature rise + resting-HR rise as physiological anchors the symptom-only model
lacks. We engineer causal (past-only) anchor features that target the biphasic shift and
measure the Fertility F1 lift over a multimodal base model.

All features are within-subject and past-only (shift(1)) so nothing leaks across LOSO.
Runs on the 2022 master table, multimodal view (warmup dropped).
"""
from __future__ import annotations
import pathlib, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, accuracy_score
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
CLASSES = ["Menstrual", "Follicular", "Fertility", "Luteal"]

SYMPTOMS = ["flow_volume", "cramps", "bloating", "moodswing", "sorebreasts", "fatigue",
            "headaches", "appetite", "stress"]
WEARABLE = ["rhr_value", "sleep_resting_heart_rate", "hr_bpm_mean", "hrv_rmssd_mean",
            "resp_full_sleep_breathing_rate", "temp_nightly_temperature",
            "wtemp_temperature_diff_from_baseline", "stress_stress_score",
            "sleep_overall_score", "cgm_glucose_value_mean"]
GEN = {"Not at all": 0, "Very Low": 1, "Very Low/Little": 1, "Low": 2, "Moderate": 3,
       "High": 4, "Very High": 5, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
FV = {"Not at all": 0, "Spotting / Very Light": 1, "Light": 2, "Somewhat Light": 3,
      "Moderate": 4, "Somewhat Heavy": 5, "Heavy": 6, "Very Heavy": 7}


def base_features(df):
    df = df.sort_values(["id", "day_in_study"]).reset_index(drop=True)
    for c in SYMPTOMS:
        m = FV if c == "flow_volume" else GEN
        df[c] = pd.to_numeric(df[c].astype("object").map(m), errors="coerce")
    feats = list(SYMPTOMS) + [c for c in WEARABLE if c in df.columns]
    g = df.groupby("id", sort=False)
    new = []
    for c in feats:
        for w in (3, 5, 7):
            df[f"{c}_rm{w}"] = g[c].transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
            df[f"{c}_rs{w}"] = g[c].transform(lambda s, w=w: s.rolling(w, min_periods=1).std().fillna(0))
            new += [f"{c}_rm{w}", f"{c}_rs{w}"]
    df["days_since_flow"] = df["days_since_flow"].astype(float)
    return df, feats + new + ["days_since_flow"]


def anchor_features(df):
    """Causal (past-only) ovulation-shift anchors from temperature and resting HR."""
    g = df.groupby("id", sort=False)
    cols = []

    def trail_base(c, w=10):
        return g[c].transform(lambda s: s.shift(1).rolling(w, min_periods=3).mean())

    for c, temp in [("temp_nightly_temperature", True), ("rhr_value", False),
                    ("sleep_resting_heart_rate", False),
                    ("wtemp_temperature_diff_from_baseline", True)]:
        if c not in df.columns:
            continue
        df[f"{c}_anchor_dev"] = df[c] - trail_base(c)                 # rise vs trailing baseline
        df[f"{c}_anchor_shift3"] = df[c] - g[c].transform(            # sudden jump vs last 3d
            lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        df[f"{c}_anchor_slope"] = g[c].transform(lambda s: s.diff())  # day-over-day rate
        cols += [f"{c}_anchor_dev", f"{c}_anchor_shift3", f"{c}_anchor_slope"]
    return df, cols


def loso(df, feats):
    X, y, gr = df[feats], df["phase"], df["id"]
    yi = np.array([CLASSES.index(p) for p in y])
    logo = LeaveOneGroupOut()
    oof = np.zeros(len(df), dtype=int)
    for tr, te in logo.split(X, yi, gr):
        m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=6,
                               loss_function="MultiClass", auto_class_weights="Balanced",
                               random_seed=42, verbose=False)
        m.fit(Pool(X.iloc[tr], yi[tr]))
        oof[te] = m.predict(Pool(X.iloc[te])).ravel().astype(int)
    mf1 = f1_score(yi, oof, average="macro")
    acc = accuracy_score(yi, oof)
    per = {CLASSES[k]: f1_score(yi == k, oof == k) for k in range(4)}
    return mf1, acc, per


def main():
    df = pd.read_parquet(PROC / "mcphases_master_2022.parquet")
    df = df[~df["is_warmup"] & df["phase"].isin(CLASSES)].copy()   # multimodal view
    print(f"rows={len(df)} subjects={df.id.nunique()}")

    df, base = base_features(df)
    df, anch = anchor_features(df)

    print("\n[base = symptoms + wearables + rolling]")
    mf1, acc, per = loso(df, base)
    print(f"  macroF1={mf1:.3f} acc={acc:.3f} | " + " ".join(f"{k}={v:.3f}" for k, v in per.items()))

    print("\n[base + ovulation anchor]")
    mf1a, acca, pera = loso(df, base + anch)
    print(f"  macroF1={mf1a:.3f} acc={acca:.3f} | " + " ".join(f"{k}={v:.3f}" for k, v in pera.items()))

    print(f"\n>>> delta macroF1 = {mf1a - mf1:+.3f} | delta Fertility F1 = {pera['Fertility'] - per['Fertility']:+.3f}")


if __name__ == "__main__":
    main()

"""Attempt 2 — does the "variability > absolute" law extend to wearables?

The SOTA paper's #1 finding: symptom rolling *standard deviation* dominates (45% of
importance), because variability is baseline-invariant and generalizes across subjects,
whereas absolute levels don't. We test whether the SAME holds for WEARABLES.

Design (wearable channels only, to isolate the effect):
  A  absolute : raw + rolling MEAN (5/7/14)
  B  variability : raw + rolling STD (5/7/14)
  C  both
If B >= A, variability carries the transferable signal for wearables too. We also report
the SHAP importance share of the std features in the full model.
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
WEARABLE = ["rhr_value", "sleep_resting_heart_rate", "hr_bpm_mean", "hrv_rmssd_mean",
            "resp_full_sleep_breathing_rate", "temp_nightly_temperature",
            "wtemp_temperature_diff_from_baseline", "stress_stress_score",
            "sleep_overall_score", "cgm_glucose_value_mean"]
WINDOWS = (5, 7, 14)


def build(df):
    df = df.sort_values(["id", "day_in_study"]).reset_index(drop=True)
    g = df.groupby("id", sort=False)
    raw, means, stds = [], [], []
    for c in [w for w in WEARABLE if w in df.columns]:
        raw.append(c)
        for w in WINDOWS:
            mm, ss = f"{c}_rm{w}", f"{c}_rs{w}"
            df[mm] = g[c].transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
            df[ss] = g[c].transform(lambda s, w=w: s.rolling(w, min_periods=1).std().fillna(0))
            means.append(mm); stds.append(ss)
    return df, raw, means, stds


def loso(df, feats):
    yi = np.array([CLASSES.index(p) for p in df["phase"]])
    X, gr = df[feats], df["id"]
    oof = np.zeros(len(df), dtype=int)
    for tr, te in LeaveOneGroupOut().split(X, yi, gr):
        m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=6,
                               loss_function="MultiClass", auto_class_weights="Balanced",
                               random_seed=42, verbose=False)
        m.fit(Pool(X.iloc[tr], yi[tr]))
        oof[te] = m.predict(Pool(X.iloc[te])).ravel().astype(int)
    return f1_score(yi, oof, average="macro"), accuracy_score(yi, oof)


def main():
    df = pd.read_parquet(PROC / "mcphases_master_2022.parquet")
    df = df[~df["is_warmup"] & df["phase"].isin(CLASSES)].copy()
    df, raw, means, stds = build(df)
    print(f"rows={len(df)} | wearables={len(raw)} | mean-feats={len(means)} std-feats={len(stds)}\n")

    for name, feats in [("A absolute (raw+mean)", raw + means),
                        ("B variability (raw+std)", raw + stds),
                        ("C both", raw + means + stds)]:
        mf1, acc = loso(df, feats)
        print(f"  {name:26} macroF1={mf1:.3f} acc={acc:.3f}")

    # SHAP importance share of std features in the 'both' model
    yi = np.array([CLASSES.index(p) for p in df["phase"]])
    feats = raw + means + stds
    m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=6,
                           loss_function="MultiClass", auto_class_weights="Balanced",
                           random_seed=42, verbose=False)
    m.fit(Pool(df[feats], yi))
    sv = m.get_feature_importance(Pool(df[feats], yi), type="ShapValues")
    imp = np.abs(sv[:, :, :-1]).mean(axis=(0, 1))
    s = pd.Series(imp, index=feats)
    tot = s.sum()
    share_std = s[stds].sum() / tot
    share_mean = s[means].sum() / tot
    print(f"\nSHAP importance share (wearables):  std={share_std:.1%}  mean={share_mean:.1%}  "
          f"raw={s[raw].sum()/tot:.1%}   (paper: symptom std ~45%)")
    print("top-8 features:")
    for k, v in s.sort_values(ascending=False).head(8).items():
        print(f"    {k:38} {v/tot:.1%}")


if __name__ == "__main__":
    main()

"""Modality ablation — the wearable contribution the SOTA paper never measured.

The paper is symptom-only (0.662). It never reports a wearable+symptom score; multimodal
fusion is its stated future work. Here we measure, under ONE identical 2022-aligned,
leakage-free LOSO pipeline, the pure lift of adding wearables:

  symptoms_only : self-report symptoms + rolling + flow-derived cycle position
  wearables_only: Fitbit/CGM channels + rolling + ovulation anchors (NO self-report at all)
  all           : symptoms + wearables + anchors + cycle position

Same CatBoost + per-fold HSMM for each. This is the controlled evidence for "wearables add
X" — unlike comparing our multimodal number to the paper's symptom-only number across papers.
"""
from __future__ import annotations
import pathlib, sys, re, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, accuracy_score
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from attempt1_ovulation import base_features, anchor_features, CLASSES, PROC, SYMPTOMS, WEARABLE
from attempt3_hsmm import learn_durations, seg_viterbi
from attempt5_fertility_detector import days_since_bleed


def base_sig(c):
    return re.sub(r"_(rm|rs)\d+$", "", c)


def main():
    df = pd.read_parquet(PROC / "mcphases_master_2022.parquet")
    df = df[~df["is_warmup"] & df["phase"].isin(CLASSES)].copy()
    df, base = base_features(df)
    df, anch = anchor_features(df)
    df["days_since_bleed"] = days_since_bleed(df)
    base = [f for f in base if f != "days_since_flow"]

    sym_feats = [c for c in base if base_sig(c) in SYMPTOMS] + ["days_since_bleed"]
    wear_feats = [c for c in base if base_sig(c) in WEARABLE] + anch          # no self-report
    all_feats = base + anch + ["days_since_bleed"]

    yi = np.array([CLASSES.index(p) for p in df["phase"]])
    g = df["id"].to_numpy(); days = df["day_in_study"].to_numpy()

    def loso_hsmm(feats):
        P = np.zeros((len(df), 4))
        for tr, te in LeaveOneGroupOut().split(df[feats], yi, g):
            m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=6,
                                   loss_function="MultiClass", auto_class_weights="Balanced",
                                   random_seed=42, verbose=False)
            m.fit(Pool(df[feats].iloc[tr], yi[tr]))
            p = m.predict_proba(Pool(df[feats].iloc[te]))
            P[te] = p[:, [list(m.classes_).index(k) for k in range(4)]]
        out = np.zeros(len(df), int); le = np.log(np.clip(P, 1e-9, 1))
        for s in np.unique(g):
            te = g == s
            mu, sd = learn_durations(yi[~te], g[~te], days[~te])
            idx = np.where(te)[0]; order = idx[np.argsort(days[idx])]
            out[order] = seg_viterbi(le[order], mu, sd)
        return P.argmax(1), out

    print(f"rows={len(df)} | sym={len(sym_feats)} wear={len(wear_feats)} all={len(all_feats)} feats\n")
    rows = []
    for name, feats in [("symptoms_only", sym_feats),
                        ("wearables_only", wear_feats),
                        ("all (multimodal)", all_feats)]:
        raw, hs = loso_hsmm(feats)
        per = {CLASSES[k]: f1_score(yi == k, hs == k) for k in range(4)}
        mf1 = f1_score(yi, hs, average="macro")
        print(f"  {name:18} +HSMM  macroF1={mf1:.3f} acc={accuracy_score(yi,hs):.3f} | "
              + " ".join(f"{c[:4]}={v:.3f}" for c, v in per.items()))
        rows.append((name, mf1, per["Fertility"]))
    sym = dict((r[0], r[1]) for r in rows)
    print(f"\n>>> wearable lift over symptoms-only: macroF1 {sym['all (multimodal)']-sym['symptoms_only']:+.3f}")
    print("(paper symptom-only reference = 0.662)")


if __name__ == "__main__":
    main()

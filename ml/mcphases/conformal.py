"""Split conformal prediction over the LOSO out-of-fold probabilities.

For each test subject, the conformal threshold is calibrated on the OTHER 41 subjects'
out-of-fold nonconformity scores (LOSO-safe: never uses the test subject's own days).
Mondrian / class-conditional: a separate quantile per phase, so coverage holds
separately for each phase (and, by extension, for regular vs irregular cycles).

Score = LAC (1 - prob_trueclass). Prediction set = {c : prob_c >= 1 - q_c}.
  singleton set  -> confident call
  empty/multi set -> no-call (abstain)

Usage:  python ml/mcphases/conformal.py --baseline B1 --alpha 0.1
"""
from __future__ import annotations
import argparse, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
CLASSES = ["Menstrual", "Follicular", "Fertility", "Luteal"]


def run(baseline: str, alpha: float = 0.1):
    df = pd.read_parquet(PROC / f"oof_{baseline}.parquet").reset_index(drop=True)
    P = df[[f"prob_{c}" for c in CLASSES]].to_numpy()
    y = df["phase"].to_numpy()
    subj = df["id"].to_numpy()
    true_idx = np.array([CLASSES.index(t) for t in y])
    nonconf = 1.0 - P[np.arange(len(df)), true_idx]      # LAC score on the true class

    sets = [None] * len(df)
    for s in np.unique(subj):
        te = subj == s
        cal = ~te
        # Mondrian: per-class calibration quantile from the other subjects
        q = {}
        for k, c in enumerate(CLASSES):
            mask = cal & (true_idx == k)
            scores = nonconf[mask]
            n = len(scores)
            if n == 0:
                q[k] = 1.0
                continue
            lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)   # finite-sample correction
            q[k] = np.quantile(scores, lvl, method="higher")
        for i in np.where(te)[0]:
            sets[i] = [c for k, c in enumerate(CLASSES) if (1.0 - P[i, k]) <= q[k]]

    set_size = np.array([len(s) for s in sets])
    covered = np.array([y[i] in sets[i] for i in range(len(df))])
    singleton = set_size == 1
    no_call = set_size != 1                                 # empty or multi -> abstain
    conf_correct = np.array([singleton[i] and sets[i][0] == y[i] for i in range(len(df))])

    print(f"[{baseline}] split conformal  (target coverage = {1-alpha:.0%}, Mondrian per-phase)")
    print(f"  empirical coverage (true phase in set) : {covered.mean():.3f}")
    print(f"  mean prediction-set size               : {set_size.mean():.2f}")
    print(f"  confident-call rate (singleton set)    : {singleton.mean():.3f}")
    print(f"  no-call / abstain rate                 : {no_call.mean():.3f}")
    print(f"  accuracy ON confident calls            : {conf_correct.sum()/max(singleton.sum(),1):.3f}")
    print("  per-phase coverage:")
    for k, c in enumerate(CLASSES):
        mk = true_idx == k
        print(f"    {c:11} coverage={covered[mk].mean():.3f}  n={mk.sum()}")

    df["set_size"] = set_size
    df["covered"] = covered
    df["no_call"] = no_call
    df["pred_set"] = ["|".join(s) for s in sets]
    df.to_parquet(PROC / f"oof_{baseline}_conformal.parquet", index=False)
    print(f"saved -> {PROC / f'oof_{baseline}_conformal.parquet'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="B1")
    ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args()
    run(a.baseline, a.alpha)

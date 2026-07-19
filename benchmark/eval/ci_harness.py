"""Subject-bootstrap confidence intervals for the mcPHASES phase benchmark.

At N=42 subjects a single macro-F1 point is not enough to claim you beat (or match)
another method — the run-to-run and between-subject variance is ~0.01-0.02. This tool
resamples SUBJECTS with replacement and reports a 95% CI, so a comparison to the
published SOTA (0.662 / 0.676) is statistically honest.

Input: an out-of-fold predictions table (parquet/csv) with columns
  id, phase (true label), pred (predicted label).
Model-agnostic: works on any method's OOF file.

Usage:  python benchmark/eval/ci_harness.py --oof benchmark/results/oof_reference.parquet
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

CLASSES = ["Menstrual", "Follicular", "Fertility", "Luteal"]
SOTA = {"macro_f1": 0.662, "accuracy": 0.676}


def subject_bootstrap(df: pd.DataFrame, B: int = 2000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    subs = df["id"].unique()
    # precompute per-subject (true, pred) arrays for speed
    grp = {s: (g["phase"].to_numpy(), g["pred"].to_numpy())
           for s, g in df.groupby("id")}
    keys = [f"f1_{c}" for c in CLASSES]
    acc = {"macro_f1": [], "accuracy": [], **{k: [] for k in keys}}
    for _ in range(B):
        samp = rng.choice(subs, len(subs), replace=True)
        y = np.concatenate([grp[s][0] for s in samp])
        p = np.concatenate([grp[s][1] for s in samp])
        acc["macro_f1"].append(f1_score(y, p, average="macro", labels=CLASSES))
        acc["accuracy"].append(accuracy_score(y, p))
        for c in CLASSES:
            acc[f"f1_{c}"].append(f1_score(y == c, p == c))

    def ci(a):
        a = np.asarray(a)
        return float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
    return {k: ci(v) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True, help="OOF parquet/csv with id, phase, pred")
    ap.add_argument("-B", type=int, default=2000)
    a = ap.parse_args()
    df = pd.read_parquet(a.oof) if a.oof.endswith(".parquet") else pd.read_csv(a.oof)

    pt_mf1 = f1_score(df["phase"], df["pred"], average="macro", labels=CLASSES)
    pt_acc = accuracy_score(df["phase"], df["pred"])
    res = subject_bootstrap(df, a.B)

    print(f"OOF: {len(df)} rows, {df['id'].nunique()} subjects, B={a.B} subject-bootstrap")
    print(f"point estimate: macro-F1={pt_mf1:.3f}  acc={pt_acc:.3f}")
    print(f"SOTA reference: macro-F1={SOTA['macro_f1']:.3f}  acc={SOTA['accuracy']:.3f}\n")
    for k, (m, lo, hi) in res.items():
        print(f"  {k:14} {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    lo_mf1 = res["macro_f1"][1]; hi_mf1 = res["macro_f1"][2]
    verdict = ("CI contains SOTA -> statistically ON PAR"
               if lo_mf1 <= SOTA["macro_f1"] <= hi_mf1 else
               "above SOTA" if lo_mf1 > SOTA["macro_f1"] else "below SOTA")
    print(f"\nverdict vs SOTA macro-F1 {SOTA['macro_f1']}: {verdict}")


if __name__ == "__main__":
    main()

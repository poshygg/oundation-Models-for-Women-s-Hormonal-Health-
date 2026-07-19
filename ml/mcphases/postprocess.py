"""Temporal smoothing (HSMM-style Viterbi) over per-day phase probabilities.

Reads oof_<baseline>.parquet (per-day class probabilities from LOSO), then decodes
each subject/interval day-sequence with a Hidden (Semi-)Markov constraint:
  - cyclic phase order  Menstrual -> Follicular -> Fertility -> Luteal -> Menstrual
  - a soft self-transition / advance prior (duration prior via self-loop probability)
This removes physiologically impossible day-to-day jumps a per-day classifier can make.

Usage:  python ml/mcphases/postprocess.py --baseline B0
"""
from __future__ import annotations
import argparse, pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
CLASSES = ["Menstrual", "Follicular", "Fertility", "Luteal"]
K = len(CLASSES)


def transition_logprob(stay: float = 0.80) -> np.ndarray:
    """Cyclic transition matrix: stay in phase, or advance to the next. Duration prior
    is encoded by `stay` (higher -> longer phases)."""
    T = np.full((K, K), 1e-6)
    for i in range(K):
        T[i, i] = stay
        T[i, (i + 1) % K] = 1.0 - stay
    return np.log(T / T.sum(1, keepdims=True))


def viterbi(log_emis: np.ndarray, log_trans: np.ndarray) -> np.ndarray:
    n = log_emis.shape[0]
    dp = np.full((n, K), -np.inf)
    bp = np.zeros((n, K), dtype=int)
    dp[0] = log_emis[0] + np.log(1.0 / K)
    for t in range(1, n):
        for j in range(K):
            scores = dp[t - 1] + log_trans[:, j]
            bp[t, j] = scores.argmax()
            dp[t, j] = scores.max() + log_emis[t, j]
    path = np.zeros(n, dtype=int)
    path[-1] = dp[-1].argmax()
    for t in range(n - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]
    return path


def run(baseline: str, stay: float = 0.80):
    df = pd.read_parquet(PROC / f"oof_{baseline}.parquet")
    prob = df[[f"prob_{c}" for c in CLASSES]].to_numpy()
    log_emis_all = np.log(np.clip(prob, 1e-9, 1))
    log_trans = transition_logprob(stay)

    smoothed = np.empty(len(df), dtype=object)
    for _, idx in df.groupby(["id", "study_interval"]).groups.items():
        idx = sorted(idx, key=lambda i: df.loc[i, "day_in_study"])
        pos = [df.index.get_loc(i) for i in idx]
        path = viterbi(log_emis_all[pos], log_trans)
        for p, s in zip(pos, path):
            smoothed[p] = CLASSES[s]

    y = df["phase"].to_numpy()
    a0, b0, f0 = accuracy_score(y, df["pred"]), balanced_accuracy_score(y, df["pred"]), f1_score(y, df["pred"], average="macro", labels=CLASSES)
    a1, b1, f1_ = accuracy_score(y, smoothed), balanced_accuracy_score(y, smoothed), f1_score(y, smoothed, average="macro", labels=CLASSES)
    print(f"[{baseline}] before smoothing:  acc={a0:.3f}  bacc={b0:.3f}  macroF1={f0:.3f}")
    print(f"[{baseline}] after  HSMM  (stay={stay}):  acc={a1:.3f}  bacc={b1:.3f}  macroF1={f1_:.3f}")
    df["pred_hsmm"] = smoothed
    df.to_parquet(PROC / f"oof_{baseline}_hsmm.parquet", index=False)
    print(f"saved -> {PROC / f'oof_{baseline}_hsmm.parquet'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="B0")
    ap.add_argument("--stay", type=float, default=0.80)
    run(ap.parse_args().baseline, ap.parse_args().stay)

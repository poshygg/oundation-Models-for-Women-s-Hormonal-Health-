"""Explicit-duration HSMM smoothing with learned Gaussian duration priors.

Upgrade over postprocess.py (fixed self-transition): each phase has its own duration
distribution Gaussian(mu, sd) fit from the true phase run-lengths (a population-level
biological prior, e.g. luteal ~= 12 days -- not per-sample label leakage). Decoding is a
segmental (explicit-duration) Viterbi over the cyclic phase order, with the per-day
classifier probabilities as emissions.

Usage:  python ml/mcphases/postprocess_hsmm.py --baseline B1
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


def learn_durations(df: pd.DataFrame):
    """Fit Gaussian(mu, sd) per phase from true run-lengths across subject/interval series."""
    runs = {k: [] for k in range(K)}
    idx = {c: k for k, c in enumerate(CLASSES)}
    for _, sub in df.groupby(["id", "study_interval"]):
        seq = [idx[p] for p in sub.sort_values("day_in_study")["phase"]]
        if not seq:
            continue
        cur, length = seq[0], 1
        for s in seq[1:]:
            if s == cur:
                length += 1
            else:
                runs[cur].append(length); cur, length = s, 1
        runs[cur].append(length)
    mu = np.array([np.mean(runs[k]) if runs[k] else 7.0 for k in range(K)])
    sd = np.array([max(np.std(runs[k]), 1.5) if runs[k] else 3.0 for k in range(K)])
    return mu, sd


def dur_logpmf(mu, sd, Dmax):
    d = np.arange(1, Dmax + 1)[:, None]                 # (Dmax,1)
    lp = -0.5 * ((d - mu) / sd) ** 2 - np.log(sd)       # (Dmax,K) unnormalised Gaussian
    return lp                                            # index [d-1, j]


def segmental_viterbi(log_emis, dlogp, Dmax):
    """Explicit-duration Viterbi over cyclic states (j reachable only from (j-1)%K)."""
    T = log_emis.shape[0]
    cum = np.vstack([np.zeros(K), np.cumsum(log_emis, axis=0)])   # (T+1,K) prefix sums
    NEG = -1e18
    dp = np.full((T + 1, K), NEG)
    bp = np.full((T + 1, K, 2), -1, dtype=int)          # backpointer (prev_t, prev_state)
    for t in range(1, T + 1):
        for j in range(K):
            i = (j - 1) % K                             # only predecessor in cyclic order
            best, arg = NEG, (-1, -1)
            for d in range(1, min(Dmax, t) + 1):
                seg = cum[t, j] - cum[t - d, j] + dlogp[d - 1, j]
                if t - d == 0:                          # first segment: any start state
                    cand = seg
                    prev = (0, -1)
                else:
                    cand = dp[t - d, i] + seg
                    prev = (t - d, i)
                if cand > best:
                    best, arg = cand, prev
            dp[t, j] = best; bp[t, j] = arg
    # backtrack
    path = np.zeros(T, dtype=int)
    t, j = T, int(np.argmax(dp[T]))
    while t > 0:
        pt, pi = bp[t, j]
        path[pt:t] = j
        t, j = pt, pi if pi >= 0 else j
        if pi < 0:
            break
    return path


def run(baseline: str, Dmax: int = 45):
    df = pd.read_parquet(PROC / f"oof_{baseline}.parquet").reset_index(drop=True)
    mu, sd = learn_durations(df)
    print("learned phase durations (days):",
          {CLASSES[k]: f"{mu[k]:.1f}±{sd[k]:.1f}" for k in range(K)})
    dlogp = dur_logpmf(mu, sd, Dmax)

    prob = df[[f"prob_{c}" for c in CLASSES]].to_numpy()
    log_emis_all = np.log(np.clip(prob, 1e-9, 1))
    smoothed = np.empty(len(df), dtype=object)
    for _, sub in df.groupby(["id", "study_interval"]):
        order = sub.sort_values("day_in_study").index.to_list()
        pos = [df.index.get_loc(i) for i in order]
        path = segmental_viterbi(log_emis_all[pos], dlogp, Dmax)
        for p, s in zip(pos, path):
            smoothed[p] = CLASSES[s]

    y = df["phase"].to_numpy()
    def rep(tag, pred):
        print(f"[{baseline}] {tag:20} acc={accuracy_score(y,pred):.3f} "
              f"bacc={balanced_accuracy_score(y,pred):.3f} "
              f"macroF1={f1_score(y,pred,average='macro',labels=CLASSES):.3f}")
    rep("raw classifier", df["pred"])
    rep("HSMM(duration)", smoothed)
    df["pred_hsmm_dur"] = smoothed
    df.to_parquet(PROC / f"oof_{baseline}_hsmmdur.parquet", index=False)
    print(f"saved -> {PROC / f'oof_{baseline}_hsmmdur.parquet'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="B1")
    ap.add_argument("--Dmax", type=int, default=45)
    a = ap.parse_args()
    run(a.baseline, a.Dmax)

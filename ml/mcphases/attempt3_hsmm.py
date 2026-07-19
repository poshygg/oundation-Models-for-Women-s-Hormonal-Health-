"""HSMM proper on the aligned 2022 data — the paper's +0.03 lever.

Take the base+anchor CatBoost (macro-F1 ~0.62 on the 2022 master table), collect its
LOSO out-of-fold day-probabilities, then decode each subject's day-sequence with an
explicit-duration HSMM: cyclic phase order + learned Gaussian duration priors, with a
small skip probability so anovulatory cycles can bypass the Fertility window.

Durations are learned from the TRAINING subjects of each fold (leakage-free).
"""
from __future__ import annotations
import pathlib, warnings, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, accuracy_score
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from attempt1_ovulation import base_features, anchor_features, CLASSES, PROC

K = len(CLASSES)
FERT = CLASSES.index("Fertility")


def learn_durations(y_idx, groups, days):
    runs = {k: [] for k in range(K)}
    df = pd.DataFrame({"y": y_idx, "g": groups, "d": days}).sort_values(["g", "d"])
    for _, sub in df.groupby("g"):
        seq = sub["y"].to_numpy()
        cur, ln = seq[0], 1
        for s in seq[1:]:
            if s == cur:
                ln += 1
            else:
                runs[cur].append(ln); cur, ln = s, 1
        runs[cur].append(ln)
    mu = np.array([np.mean(runs[k]) if runs[k] else 7.0 for k in range(K)])
    sd = np.array([max(np.std(runs[k]), 1.5) if runs[k] else 3.0 for k in range(K)])
    return mu, sd


def seg_viterbi(log_emis, mu, sd, Dmax=45, skip_prob=0.02):
    T = log_emis.shape[0]
    cum = np.vstack([np.zeros(K), np.cumsum(log_emis, axis=0)])
    d = np.arange(1, Dmax + 1)[:, None]
    dlp = -0.5 * ((d - mu) / sd) ** 2 - np.log(sd)          # (Dmax,K)
    NEG = -1e18
    dp = np.full((T + 1, K), NEG); bp = np.full((T + 1, K, 2), -1, int)
    log_skip, log_noskip = np.log(skip_prob), np.log(1 - skip_prob)
    for t in range(1, T + 1):
        for j in range(K):
            preds = [((j - 1) % K, log_noskip)]
            if j == (FERT + 1) % K:                          # Luteal can follow Follicular (skip Fertility)
                preds.append(((FERT - 1) % K, log_skip))
            best, arg = NEG, (-1, -1)
            for dd in range(1, min(Dmax, t) + 1):
                seg = cum[t, j] - cum[t - dd, j] + dlp[dd - 1, j]
                if t - dd == 0:
                    cand, prev = seg, (0, -1)
                    if cand > best:
                        best, arg = cand, prev
                else:
                    for i, tr in preds:
                        cand = dp[t - dd, i] + seg + tr
                        if cand > best:
                            best, arg = cand, (t - dd, i)
            dp[t, j], bp[t, j] = best, arg
    path = np.zeros(T, int); t, j = T, int(dp[T].argmax())
    while t > 0:
        pt, pi = bp[t, j]; path[pt:t] = j
        if pi < 0:
            break
        t, j = pt, pi
    return path


def main():
    df = pd.read_parquet(PROC / "mcphases_master_2022.parquet")
    df = df[~df["is_warmup"] & df["phase"].isin(CLASSES)].copy()
    df, base = base_features(df); df, anch = anchor_features(df)
    feats = base + anch
    yi = np.array([CLASSES.index(p) for p in df["phase"]])
    g = df["id"].to_numpy(); days = df["day_in_study"].to_numpy()

    proba = np.zeros((len(df), K))
    for tr, te in LeaveOneGroupOut().split(df[feats], yi, g):
        m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=6,
                               loss_function="MultiClass", auto_class_weights="Balanced",
                               random_seed=42, verbose=False)
        m.fit(Pool(df[feats].iloc[tr], yi[tr]))
        p = m.predict_proba(Pool(df[feats].iloc[te]))
        proba[te] = p[:, [list(m.classes_).index(k) for k in range(K)]]

    raw_pred = proba.argmax(1)
    # HSMM decode per subject; durations learned from OTHER subjects (leakage-free)
    smoothed = np.zeros(len(df), int)
    log_emis = np.log(np.clip(proba, 1e-9, 1))
    for s in np.unique(g):
        te = g == s
        mu, sd = learn_durations(yi[~te], g[~te], days[~te])
        idx = np.where(te)[0]; order = idx[np.argsort(days[idx])]
        path = seg_viterbi(log_emis[order], mu, sd)
        smoothed[order] = path

    def rep(tag, pred):
        per = " ".join(f"{CLASSES[k][:4]}={f1_score(yi==k,pred==k):.3f}" for k in range(K))
        print(f"  {tag:16} macroF1={f1_score(yi,pred,average='macro'):.3f} "
              f"acc={accuracy_score(yi,pred):.3f} | {per}")
    print(f"rows={len(df)} subjects={df.id.nunique()}")
    rep("base (raw)", raw_pred)
    rep("+ HSMM", smoothed)
    print("(paper: CatBoost 0.629 -> +HSMM 0.662)")


if __name__ == "__main__":
    main()

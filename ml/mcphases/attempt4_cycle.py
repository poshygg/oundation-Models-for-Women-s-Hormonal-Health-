"""sin/cos cyclical encoding of cycle position (the paper's missing-for-us feature).

Cycle position is periodic (day 28 -> day 1 wraps), so a linear days-since-flow counter
tells the model that end-of-cycle and start-of-cycle are far apart when they are adjacent.
We encode it on a circle: angle = 2*pi * days_since_flow / cycle_length, then sin/cos, plus
a normalized cycle-progress. All derived from reported FLOW (not the label) -> no leakage;
cycle_length is each subject's own median bleed-to-bleed interval.

Pipeline: base + anchor + cycle -> LOSO CatBoost -> explicit-duration HSMM.
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
from attempt3_hsmm import learn_durations, seg_viterbi

K = len(CLASSES)


def add_cycle_features(df, default_len=28):
    df = df.sort_values(["id", "day_in_study"]).reset_index(drop=True)
    dsf = df["days_since_flow"].to_numpy(float)
    day = df["day_in_study"].to_numpy()
    sin = np.zeros(len(df)); cos = np.zeros(len(df))
    prog = np.zeros(len(df)); clen = np.full(len(df), float(default_len))
    for _, idx in df.groupby("id", sort=False).groups.items():
        idx = list(idx); d = dsf[idx]; dd = day[idx]
        onsets = [dd[i] for i in range(len(idx)) if d[i] == 0 and (i == 0 or d[i - 1] > 0)]
        gaps = np.diff(onsets)
        L = float(np.median(gaps)) if len(gaps) >= 1 else default_len
        L = min(max(L, 20), 40)                      # clamp to plausible cycle length
        for i, ix in enumerate(idx):
            ang = 2 * np.pi * d[i] / L
            sin[ix], cos[ix] = np.sin(ang), np.cos(ang)
            prog[ix] = min(d[i] / L, 1.5)
        for ix in idx:
            clen[ix] = L
    df["cyc_sin"], df["cyc_cos"] = sin, cos
    df["cyc_progress"], df["cyc_len"] = prog, clen
    return df, ["cyc_sin", "cyc_cos", "cyc_progress", "cyc_len"]


def loso_proba(df, feats):
    yi = np.array([CLASSES.index(p) for p in df["phase"]])
    g = df["id"].to_numpy()
    proba = np.zeros((len(df), K))
    for tr, te in LeaveOneGroupOut().split(df[feats], yi, g):
        m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=6,
                               loss_function="MultiClass", auto_class_weights="Balanced",
                               random_seed=42, verbose=False)
        m.fit(Pool(df[feats].iloc[tr], yi[tr]))
        p = m.predict_proba(Pool(df[feats].iloc[te]))
        proba[te] = p[:, [list(m.classes_).index(k) for k in range(K)]]
    return proba, yi, g


def hsmm(proba, yi, g, days):
    out = np.zeros(len(proba), int)
    le = np.log(np.clip(proba, 1e-9, 1))
    for s in np.unique(g):
        te = g == s
        mu, sd = learn_durations(yi[~te], g[~te], days[~te])
        idx = np.where(te)[0]; order = idx[np.argsort(days[idx])]
        out[order] = seg_viterbi(le[order], mu, sd)
    return out


def report(tag, yi, pred):
    per = " ".join(f"{CLASSES[k][:4]}={f1_score(yi==k,pred==k):.3f}" for k in range(K))
    print(f"  {tag:22} macroF1={f1_score(yi,pred,average='macro'):.3f} "
          f"acc={accuracy_score(yi,pred):.3f} | {per}")


def main():
    df = pd.read_parquet(PROC / "mcphases_master_2022.parquet")
    df = df[~df["is_warmup"] & df["phase"].isin(CLASSES)].copy()
    df, base = base_features(df)
    df, anch = anchor_features(df)
    df, cyc = add_cycle_features(df)
    days = df["day_in_study"].to_numpy()
    print(f"rows={len(df)} | +cycle feats: {cyc}\n")

    for tag, feats in [("base+anchor", base + anch),
                       ("base+anchor+cycle", base + anch + cyc)]:
        proba, yi, g = loso_proba(df, feats)
        report(tag, yi, proba.argmax(1))
        report(tag + " +HSMM", yi, hsmm(proba, yi, g, days))
    print("(prev best +HSMM = 0.644 ; SOTA 0.662)")


if __name__ == "__main__":
    main()

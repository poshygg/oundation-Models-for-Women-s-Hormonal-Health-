"""Cheap temporal post-processors vs HSMM: does the learned-duration HSMM actually
beat a trivial smoothing filter, or is most of its lift just local averaging?

On CatBoost OOF probabilities (base+anchor+dsb, days_since_flow dropped), per subject
in day order:
  - raw               : argmax, no smoothing
  - mode-filter(w)    : majority vote of argmax labels in a centered window
  - meanprob-filter(w): moving average of probability vectors, then argmax
  - HSMM (learned dur): the incumbent, for reference
Standalone; edits no shared pipeline file (runs alongside the Fertility work).
"""
import warnings; warnings.filterwarnings("ignore")
import sys, time
import numpy as np, pandas as pd, pathlib
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, accuracy_score
sys.path.insert(0, ".")
sys.path.insert(0, "ml/mcphases")
from model_layer_experiment import base_features, anchor_features, learn_hsmm_params, make_cat
from model_layer_dsb_ensemble import days_since_bleed
from ai.hormonal.smoothing import hsmm_smooth

PROC = pathlib.Path("data/processed")
CLASSES = ["Menstrual", "Follicular", "Fertility", "Luteal"]


def mode_filter(pred, seg, w):
    half = w // 2; out = pred.copy()
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]; p = pred[idx]; o = p.copy()
        for i in range(len(p)):
            lo, hi = max(0, i - half), min(len(p), i + half + 1)
            o[i] = np.bincount(p[lo:hi], minlength=4).argmax()
        out[idx] = o
    return out


def meanprob_filter(P, seg, w):
    half = w // 2; out = P.copy()
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]; sub = P[idx]; o = sub.copy()
        for i in range(len(sub)):
            lo, hi = max(0, i - half), min(len(sub), i + half + 1)
            o[i] = sub[lo:hi].mean(0)
        out[idx] = o
    return out.argmax(1)


def loso_proba(df, feats):
    X = df[feats]; yi = np.array([CLASSES.index(p) for p in df["phase"]]); gr = df["id"].to_numpy()
    oofp = np.zeros((len(df), 4))
    for tr, te in LeaveOneGroupOut().split(X, yi, gr):
        m = make_cat(); m.fit(X.iloc[tr], yi[tr]); oofp[te] = m.predict_proba(X.iloc[te])
    return oofp, yi


def show(tag, yi, pred):
    per = {CLASSES[k]: f1_score(yi == k, pred == k) for k in range(4)}
    print(f"  {tag:22s} macroF1={f1_score(yi, pred, average='macro'):.3f} "
          f"acc={accuracy_score(yi, pred):.3f} | Fert={per['Fertility']:.3f} "
          f"Mens={per['Menstrual']:.3f} Foll={per['Follicular']:.3f} Lute={per['Luteal']:.3f}")


def main():
    df = pd.read_parquet(PROC / "mcphases_master_2022.parquet")
    df = df[~df["is_warmup"] & df["phase"].isin(CLASSES)].copy()
    df, base = base_features(df); df, anch = anchor_features(df)
    df["days_since_bleed"] = days_since_bleed(df)
    base = [f for f in base if f != "days_since_flow"]
    feats = base + anch + ["days_since_bleed"]
    seg = df["id"].to_numpy()
    mean, sd, _ = learn_hsmm_params(df)

    t = time.time(); oofp, yi = loso_proba(df, feats)
    print(f"rows={len(df)} feats={len(feats)}  CatBoost LOSO ({time.time()-t:.0f}s)\n")

    show("raw (no smoothing)", yi, oofp.argmax(1)); print()
    for w in (3, 5, 7):
        show(f"mode-filter w={w}", yi, mode_filter(oofp.argmax(1), seg, w))
    print()
    for w in (3, 5, 7):
        show(f"meanprob-filter w={w}", yi, meanprob_filter(oofp, seg, w))
    print()
    show("HSMM(learned dur)", yi, hsmm_smooth(oofp, seg, mean_dur=mean, sd_dur=sd, skip_prob=0.02))


if __name__ == "__main__":
    main()

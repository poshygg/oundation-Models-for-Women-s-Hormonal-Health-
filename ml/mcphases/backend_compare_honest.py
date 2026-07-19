"""Backend comparison on the current corrected setting (master_2022, dsb, per-fold HSMM).

Single-stage 4-class classifier swapped across CatBoost / XGBoost / LightGBM under one
identical LOSO split and one identical feature matrix, so any difference is the backend's,
not the pipeline's. Balancing is identical across backends (balanced sample weights).
Cycle position comes from self-reported flow only (no label). No tuning on the test split.

Run from repo root:  python ml/mcphases/backend_compare_honest.py
"""
from __future__ import annotations
import pathlib, sys, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "mcphases"))
from attempt1_ovulation import base_features, anchor_features, CLASSES  # noqa: E402
from attempt3_hsmm import learn_durations, seg_viterbi                   # noqa: E402

PROC = ROOT / "data" / "processed"
BLEED_MIN_SEV, BLEED_MIN_GAP = 2.0, 10
NC = len(CLASSES)


def days_since_bleed(df):
    flow = df["flow_volume"].to_numpy(float)
    out = np.full(len(df), np.nan)
    for _, idx in df.groupby("id", sort=False).groups.items():
        rows = list(idx); onset, last = None, -10**9
        for i, r in enumerate(rows):
            bleed = (not np.isnan(flow[r])) and flow[r] >= BLEED_MIN_SEV
            if bleed and (i - last) >= BLEED_MIN_GAP:
                onset = i
            if bleed:
                last = i
            out[r] = (i - onset) if onset is not None else np.nan
    return out


def make_model(name, y_tr):
    """Return (fit_fn, proba_fn) for the backend; balancing identical across all three."""
    sw = compute_sample_weight("balanced", y_tr)
    if name == "catboost":
        from catboost import CatBoostClassifier, Pool
        m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=6,
                               loss_function="MultiClass", random_seed=42, verbose=False)
        def fit(X): m.fit(Pool(X, y_tr, weight=sw))
        def proba(X):
            p = m.predict_proba(Pool(X))
            order = [list(m.classes_).index(k) for k in range(NC)]
            return p[:, order]
        return fit, proba
    if name == "xgboost":
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, reg_lambda=6,
                          objective="multi:softprob", num_class=NC, tree_method="hist",
                          random_state=42, verbosity=0)
        def fit(X): m.fit(X, y_tr, sample_weight=sw)
        def proba(X): return m.predict_proba(X)
        return fit, proba
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(n_estimators=500, max_depth=6, num_leaves=31, learning_rate=0.05,
                           reg_lambda=6, objective="multiclass", num_class=NC,
                           random_state=42, verbose=-1)
        def fit(X): m.fit(X, y_tr, sample_weight=sw)
        def proba(X): return m.predict_proba(X)
        return fit, proba
    raise ValueError(name)


def main():
    df = pd.read_parquet(PROC / "mcphases_master_2022.parquet")
    df = df[~df["is_warmup"] & df["phase"].isin(CLASSES)].copy()
    df, base = base_features(df)
    df, anch = anchor_features(df)
    df["days_since_bleed"] = days_since_bleed(df)
    base = [f for f in base if f != "days_since_flow"]
    feats = base + anch + ["days_since_bleed"]

    yi = np.array([CLASSES.index(p) for p in df["phase"]])
    g = df["id"].to_numpy(); days = df["day_in_study"].to_numpy()
    X = df[feats]
    print(f"rows={len(df)}  subjects={len(np.unique(g))}  features={len(feats)}\n")

    def hsmm(P):
        out = np.zeros(len(P), int); le = np.log(np.clip(P, 1e-9, 1))
        for s in np.unique(g):
            te = g == s
            mu, sd = learn_durations(yi[~te], g[~te], days[~te])
            idx = np.where(te)[0]; order = idx[np.argsort(days[idx])]
            out[order] = seg_viterbi(le[order], mu, sd)
        return out

    print(f"{'backend':10} | {'raw macroF1':>11} {'acc':>6} | {'+HSMM macroF1':>13} {'acc':>6} | per-class F1 (+HSMM)")
    print("-" * 100)
    for name in ["catboost", "xgboost", "lightgbm"]:
        P = np.zeros((len(df), NC))
        for tr, te in LeaveOneGroupOut().split(X, yi, g):
            fit, proba = make_model(name, yi[tr])
            fit(X.iloc[tr]); P[te] = proba(X.iloc[te])
        raw = P.argmax(1); sm = hsmm(P)
        per = " ".join(f"{CLASSES[k][:4]}={f1_score(yi==k, sm==k):.3f}" for k in range(NC))
        print(f"{name:10} | {f1_score(yi,raw,average='macro'):>11.3f} {accuracy_score(yi,raw):>6.3f} | "
              f"{f1_score(yi,sm,average='macro'):>13.3f} {accuracy_score(yi,sm):>6.3f} | {per}")


if __name__ == "__main__":
    main()

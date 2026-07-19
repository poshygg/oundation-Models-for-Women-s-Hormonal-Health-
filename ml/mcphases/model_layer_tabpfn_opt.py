"""TabPFN v2 optimization probe: can feature selection + larger ensemble +
non-leaky class balancing close the gap to CatBoost (raw 0.621 / +HSMM 0.644)?

Levers (chosen low-cost combo):
  1. feature selection — keep CatBoost's top-K importance features (TabPFN's
     attention is diluted by the many correlated rolling mean/std columns that
     trees ignore natively).
  2. ensemble size — n_estimators 4 -> 8.
  3. class balancing — per-fold train prior correction argmax(proba / prior),
     the non-leaky analogue of CatBoost auto_class_weights="Balanced".
Reuses the exact feature construction from model_layer_experiment (no leakage,
LOSO, warmup dropped). Standalone; edits no shared pipeline file.
"""
import warnings; warnings.filterwarnings("ignore")
import pathlib, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, accuracy_score
sys.path.insert(0, ".")
sys.path.insert(0, "ml/mcphases")
from model_layer_experiment import base_features, anchor_features, learn_hsmm_params, make_cat
from ai.hormonal.smoothing import hsmm_smooth

PROC = pathlib.Path("data/processed")
CLASSES = ["Menstrual", "Follicular", "Fertility", "Luteal"]
TOPK = 50
N_EST = 8


def make_tabpfn():
    from tabpfn import TabPFNClassifier
    dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    return TabPFNClassifier(device=dev, n_estimators=N_EST, ignore_pretraining_limits=True,
                            random_state=42)


def cat_importance(df, feats, yi):
    m = make_cat(); m.fit(df[feats], yi)
    return pd.Series(m.get_feature_importance(), index=feats).sort_values(ascending=False)


def loso(df, feats, make, seg):
    """OOF probabilities + non-leaky per-fold balanced-prior predictions."""
    X = df[feats]; yi = np.array([CLASSES.index(p) for p in df["phase"]]); gr = df["id"].to_numpy()
    oofp = np.zeros((len(df), 4)); bal = np.zeros(len(df), dtype=int)
    for tr, te in LeaveOneGroupOut().split(X, yi, gr):
        m = make(); m.fit(X.iloc[tr], yi[tr])
        p = m.predict_proba(X.iloc[te]); oofp[te] = p
        prior = np.bincount(yi[tr], minlength=4) / len(tr)
        bal[te] = (p / np.clip(prior, 1e-9, None)).argmax(1)
    return oofp, bal, yi


def show(tag, yi, pred):
    per = {CLASSES[k]: f1_score(yi == k, pred == k) for k in range(4)}
    print(f"  {tag:26s} macroF1={f1_score(yi, pred, average='macro'):.3f} "
          f"acc={accuracy_score(yi, pred):.3f} | Fert={per['Fertility']:.3f} "
          f"Mens={per['Menstrual']:.3f} Foll={per['Follicular']:.3f} Lute={per['Luteal']:.3f}")


def run(df, feats, tag, seg, mean, sd):
    t = time.time()
    oofp, bal, yi = loso(df, feats, make_tabpfn, seg)
    prior_g = np.bincount(yi, minlength=4) / len(yi)
    balp = oofp / np.clip(prior_g, 1e-9, None); balp /= balp.sum(1, keepdims=True)
    print(f"[{tag}]  feats={len(feats)} n_est={N_EST}  ({time.time()-t:.0f}s)")
    show("raw", yi, oofp.argmax(1))
    show("+balanced-prior", yi, bal)
    show("+HSMM", yi, hsmm_smooth(oofp, seg, mean_dur=mean, sd_dur=sd, skip_prob=0.02))
    show("+balanced +HSMM", yi, hsmm_smooth(balp, seg, mean_dur=mean, sd_dur=sd, skip_prob=0.02))
    print()


def main():
    df = pd.read_parquet(PROC / "mcphases_master_2022.parquet")
    df = df[~df["is_warmup"] & df["phase"].isin(CLASSES)].copy()
    df, base = base_features(df); df, anch = anchor_features(df)
    feats = base + anch
    seg = df["id"].to_numpy()
    yi = np.array([CLASSES.index(p) for p in df["phase"]])
    mean, sd, _ = learn_hsmm_params(df)

    imp = cat_importance(df, feats, yi)
    top = imp.head(TOPK).index.tolist()
    print(f"rows={len(df)} subjects={df.id.nunique()} feats={len(feats)}  TOPK={TOPK}")
    print(f"top-10 CatBoost importance: {[f'{k}={v:.1f}' for k, v in imp.head(10).items()]}\n")
    print("baseline (prev run): CatBoost raw 0.621 / +HSMM 0.644 | TabPFN(146,n4) raw 0.571 / +HSMM 0.610\n")

    run(df, top, f"TabPFN top-{TOPK}", seg, mean, sd)
    run(df, feats, "TabPFN full-146", seg, mean, sd)


if __name__ == "__main__":
    main()

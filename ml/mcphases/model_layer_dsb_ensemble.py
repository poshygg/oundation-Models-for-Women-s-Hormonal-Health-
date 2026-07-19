"""days_since_bleed (leakage-free cycle position) + feature reduction + CatBoost/
TabPFN ensemble. Answers: how much do these stacked levers beat CatBoost+HSMM 0.644?

  - days_since_bleed : onset-based days since last flow EPISODE (from self-reported
    flow_volume ONLY, never the phase label). Ported from ai/hormonal/data.py; the
    stronger cycle-position feature vs the any-flow-reset days_since_flow.
  - feature reduction: TabPFN on CatBoost-importance top-K (K in {25, 30}).
  - ensemble : weighted average of CatBoost(full) + TabPFN(top-K) OOF probabilities,
    weight swept, then HSMM decode.
LOSO, warmup dropped. Standalone; edits no shared pipeline file.
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
N_EST = 8
BLEED_MIN_SEV = 2.0   # >= "Light" on the flow_volume 0-7 map (excludes spotting)
BLEED_MIN_GAP = 10    # days; two episode onsets can't be closer than this


def days_since_bleed(df):
    """Onset-based cycle position from numeric flow_volume (leakage-free)."""
    flow = df["flow_volume"].to_numpy(dtype=float)
    out = np.full(len(df), np.nan)
    for _, idx in df.groupby("id", sort=False).groups.items():
        rows = list(idx)
        onset, last = None, -(10 ** 9)
        for i, r in enumerate(rows):
            bleed = (not np.isnan(flow[r])) and flow[r] >= BLEED_MIN_SEV
            if bleed and (i - last) >= BLEED_MIN_GAP:
                onset = i
            if bleed:
                last = i
            out[r] = (i - onset) if onset is not None else np.nan
    return out


def make_tabpfn():
    from tabpfn import TabPFNClassifier
    dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    return TabPFNClassifier(device=dev, n_estimators=N_EST, ignore_pretraining_limits=True,
                            random_state=42)


def loso_proba(df, feats, make):
    X = df[feats]; yi = np.array([CLASSES.index(p) for p in df["phase"]]); gr = df["id"].to_numpy()
    oofp = np.zeros((len(df), 4))
    for tr, te in LeaveOneGroupOut().split(X, yi, gr):
        m = make(); m.fit(X.iloc[tr], yi[tr]); oofp[te] = m.predict_proba(X.iloc[te])
    return oofp, yi


def show(tag, yi, pred):
    per = {CLASSES[k]: f1_score(yi == k, pred == k) for k in range(4)}
    print(f"  {tag:30s} macroF1={f1_score(yi, pred, average='macro'):.3f} "
          f"acc={accuracy_score(yi, pred):.3f} | Fert={per['Fertility']:.3f} "
          f"Mens={per['Menstrual']:.3f} Foll={per['Follicular']:.3f} Lute={per['Luteal']:.3f}")


def main():
    df = pd.read_parquet(PROC / "mcphases_master_2022.parquet")
    df = df[~df["is_warmup"] & df["phase"].isin(CLASSES)].copy()
    df, base = base_features(df); df, anch = anchor_features(df)
    df["days_since_bleed"] = days_since_bleed(df)
    ## drop days_since_flow: same flow_volume source as days_since_bleed (redundant),
    ## keep only the stronger onset-based cycle-position transform.
    base = [f for f in base if f != "days_since_flow"]
    feats = base + anch + ["days_since_bleed"]
    seg = df["id"].to_numpy()
    mean, sd, _ = learn_hsmm_params(df)

    cb = make_cat(); yi = np.array([CLASSES.index(p) for p in df["phase"]]); cb.fit(df[feats], yi)
    imp = pd.Series(cb.get_feature_importance(), index=feats).sort_values(ascending=False)
    dsb_rank = list(imp.index).index("days_since_bleed") + 1
    print(f"rows={len(df)} subjects={df.id.nunique()} feats={len(feats)}")
    print(f"days_since_bleed importance rank = {dsb_rank}  (value={imp['days_since_bleed']:.2f})")
    print(f"top-10: {[f'{k}={v:.1f}' for k, v in imp.head(10).items()]}")
    print("baseline (no dsb): CatBoost raw 0.621 / +HSMM 0.644\n")

    ## CatBoost on full feats+dsb
    t = time.time(); cb_oof, _ = loso_proba(df, feats, make_cat)
    print(f"[CatBoost full+dsb]  ({time.time()-t:.0f}s)")
    show("raw", yi, cb_oof.argmax(1))
    show("+HSMM", yi, hsmm_smooth(cb_oof, seg, mean_dur=mean, sd_dur=sd, skip_prob=0.02)); print()

    prior_g = np.bincount(yi, minlength=4) / len(yi)
    for K in (25, 30):
        top = imp.head(K).index.tolist()
        t = time.time(); tp_oof, _ = loso_proba(df, top, make_tabpfn)
        print(f"[TabPFN top-{K}+dsb]  ({time.time()-t:.0f}s)")
        show("raw", yi, tp_oof.argmax(1))
        show("+HSMM", yi, hsmm_smooth(tp_oof, seg, mean_dur=mean, sd_dur=sd, skip_prob=0.02))
        ## ensemble weight sweep (w = CatBoost weight)
        best = None
        for w in (0.5, 0.6, 0.7):
            ens = w * cb_oof + (1 - w) * tp_oof
            p_raw = ens.argmax(1)
            p_hsmm = hsmm_smooth(ens, seg, mean_dur=mean, sd_dur=sd, skip_prob=0.02)
            f_hsmm = f1_score(yi, p_hsmm, average="macro")
            show(f"ENS w_cb={w} raw", yi, p_raw)
            show(f"ENS w_cb={w} +HSMM", yi, p_hsmm)
            if best is None or f_hsmm > best[1]:
                best = (w, f_hsmm)
        print(f"  -> best ensemble (top-{K}): w_cb={best[0]}  macroF1(+HSMM)={best[1]:.3f}\n")


if __name__ == "__main__":
    main()

"""Two-stage Fertility detector — the ovulation window is every method's worst class.

Fusing ovulation anchors into the 4-class model only moved Fertility +0.008 (attempt 1):
the ovulation signal gets buried. Here a DEDICATED binary detector ("Fertility vs rest")
focuses capacity on the biphasic temperature/HR shift, then its P(fertile) is BLENDED
into the main model's Fertility probability.

Leakage-free: both the 4-class model and the detector produce out-of-fold probabilities
under the same LOSO; we blend OOF with OOF. The blend weight is FIXED a priori at 0.5
(no sweep-on-test), so the reported number is not selection-biased.
"""
from __future__ import annotations
import pathlib, sys, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, accuracy_score
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from attempt1_ovulation import base_features, anchor_features, CLASSES, PROC
from attempt3_hsmm import learn_durations, seg_viterbi

FERT = CLASSES.index("Fertility")
BLEND_W = 0.5   # fixed a priori: equal weight to the two Fertility evidences


def days_since_bleed(df, min_sev=2.0, min_gap=10):
    flow = df["flow_volume"].to_numpy(float)
    out = np.full(len(df), np.nan)
    for _, idx in df.groupby("id", sort=False).groups.items():
        rows = list(idx); onset, last = None, -10**9
        for i, r in enumerate(rows):
            bleed = (not np.isnan(flow[r])) and flow[r] >= min_sev
            if bleed and (i - last) >= min_gap:
                onset = i
            if bleed:
                last = i
            out[r] = (i - onset) if onset is not None else np.nan
    return out


def main():
    df = pd.read_parquet(PROC / "mcphases_master_2022.parquet")
    df = df[~df["is_warmup"] & df["phase"].isin(CLASSES)].copy()
    df, base = base_features(df)
    df, anch = anchor_features(df)
    df["days_since_bleed"] = days_since_bleed(df)
    base = [f for f in base if f != "days_since_flow"]
    feats_main = base + anch + ["days_since_bleed"]
    # detector: ovulation-relevant subset (temperature/HR shift anchors + cycle position + temp/HR)
    det_pool = anch + ["days_since_bleed", "temp_nightly_temperature",
                       "wtemp_temperature_diff_from_baseline", "rhr_value",
                       "sleep_resting_heart_rate", "hr_bpm_mean", "hrv_rmssd_mean", "cramps"]
    feats_det = [f for f in det_pool if f in df.columns]

    yi = np.array([CLASSES.index(p) for p in df["phase"]])
    g = df["id"].to_numpy(); days = df["day_in_study"].to_numpy()
    fert_bin = (yi == FERT).astype(int)

    main_p = np.zeros((len(df), 4)); det_p = np.zeros(len(df))
    for tr, te in LeaveOneGroupOut().split(df, yi, g):
        m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=6,
                               loss_function="MultiClass", auto_class_weights="Balanced",
                               random_seed=42, verbose=False)
        m.fit(Pool(df[feats_main].iloc[tr], yi[tr]))
        pm = m.predict_proba(Pool(df[feats_main].iloc[te]))
        main_p[te] = pm[:, [list(m.classes_).index(k) for k in range(4)]]
        d = CatBoostClassifier(iterations=400, depth=5, learning_rate=0.05, l2_leaf_reg=6,
                               loss_function="Logloss", auto_class_weights="Balanced",
                               random_seed=42, verbose=False)
        d.fit(Pool(df[feats_det].iloc[tr], fert_bin[tr]))
        pd_ = d.predict_proba(Pool(df[feats_det].iloc[te]))
        det_p[te] = pd_[:, list(d.classes_).index(1)]

    # blend Fertility evidence at fixed w, renormalize the other 3 classes proportionally
    blended = main_p.copy()
    new_fert = BLEND_W * main_p[:, FERT] + (1 - BLEND_W) * det_p
    rest = main_p.copy(); rest[:, FERT] = 0.0
    rsum = rest.sum(1)
    scale = (1 - new_fert) / np.maximum(rsum, 1e-9)
    blended = rest * scale[:, None]
    blended[:, FERT] = new_fert

    def hsmm(P):
        out = np.zeros(len(P), int); le = np.log(np.clip(P, 1e-9, 1))
        for s in np.unique(g):
            te = g == s
            mu, sd = learn_durations(yi[~te], g[~te], days[~te])
            idx = np.where(te)[0]; order = idx[np.argsort(days[idx])]
            out[order] = seg_viterbi(le[order], mu, sd)
        return out

    def rep(tag, pred):
        per = " ".join(f"{CLASSES[k][:4]}={f1_score(yi==k,pred==k):.3f}" for k in range(4))
        print(f"  {tag:26} macroF1={f1_score(yi,pred,average='macro'):.3f} "
              f"acc={accuracy_score(yi,pred):.3f} | {per}")

    print(f"rows={len(df)} | detector feats={len(feats_det)} | blend_w={BLEND_W}")
    print(f"detector alone: Fertility-vs-rest F1={f1_score(fert_bin,(det_p>0.5).astype(int)):.3f}\n")
    rep("main (4-class)", main_p.argmax(1))
    rep("main +HSMM", hsmm(main_p))
    rep("blended", blended.argmax(1))
    rep("blended +HSMM", hsmm(blended))
    print("(reference main+HSMM = 0.654 / Fert 0.498)")

    # save blended OOF for the CI harness
    oof = df[["id", "day_in_study", "phase"]].copy()
    oof["pred"] = np.array(CLASSES)[hsmm(blended)]
    out = pathlib.Path("benchmark/results/oof_fertility_detector.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(out, index=False)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

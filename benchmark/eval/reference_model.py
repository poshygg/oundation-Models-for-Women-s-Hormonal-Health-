"""Reference model for the mcPHASES 4-phase benchmark — honest, leakage-free.

Two-stage: a 4-class CatBoost over multimodal + physiology-grounded features, plus a
dedicated binary "Fertility (ovulation window) vs rest" detector; their out-of-fold
Fertility probabilities are blended at a FIXED weight (0.5, no selection on the test
split), then decoded with an explicit-duration HSMM whose durations are learned PER FOLD.

Every feature is within-subject and past-only; the cycle-position feature comes from
self-reported flow (never the label). No hyperparameter is tuned on the reported split.

Produces:
  * benchmark/splits/loso_folds.json        — frozen LOSO subject-per-fold split
  * benchmark/results/oof_reference.parquet — out-of-fold predictions for ci_harness.py

Run from the repo root:  python benchmark/eval/reference_model.py
"""
from __future__ import annotations
import json, pathlib, sys, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, accuracy_score
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "mcphases"))
from attempt1_ovulation import base_features, anchor_features, CLASSES  # noqa: E402
from attempt3_hsmm import learn_durations, seg_viterbi                   # noqa: E402

PROC = ROOT / "data" / "processed"
BENCH = ROOT / "benchmark"
FERT = CLASSES.index("Fertility")
BLEED_MIN_SEV, BLEED_MIN_GAP = 2.0, 10
BLEND_W = 0.5   # fixed a priori — equal weight to the two Fertility evidences


def days_since_bleed(df):
    """Onset-based cycle position from self-reported flow_volume only (no label)."""
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


def main():
    df = pd.read_parquet(PROC / "mcphases_master_2022.parquet")
    df = df[~df["is_warmup"] & df["phase"].isin(CLASSES)].copy()
    df, base = base_features(df)
    df, anch = anchor_features(df)
    df["days_since_bleed"] = days_since_bleed(df)
    base = [f for f in base if f != "days_since_flow"]              # redundant w/ dsb
    feats = base + anch + ["days_since_bleed"]
    det_pool = anch + ["days_since_bleed", "temp_nightly_temperature",
                       "wtemp_temperature_diff_from_baseline", "rhr_value",
                       "sleep_resting_heart_rate", "hr_bpm_mean", "hrv_rmssd_mean", "cramps"]
    feats_det = [f for f in det_pool if f in df.columns]

    yi = np.array([CLASSES.index(p) for p in df["phase"]])
    g = df["id"].to_numpy(); days = df["day_in_study"].to_numpy()
    fert_bin = (yi == FERT).astype(int)

    main_p = np.zeros((len(df), len(CLASSES))); det_p = np.zeros(len(df)); folds = []
    for tr, te in LeaveOneGroupOut().split(df[feats], yi, g):
        m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=6,
                               loss_function="MultiClass", auto_class_weights="Balanced",
                               random_seed=42, verbose=False)
        m.fit(Pool(df[feats].iloc[tr], yi[tr]))
        pm = m.predict_proba(Pool(df[feats].iloc[te]))
        main_p[te] = pm[:, [list(m.classes_).index(k) for k in range(len(CLASSES))]]
        d = CatBoostClassifier(iterations=400, depth=5, learning_rate=0.05, l2_leaf_reg=6,
                               loss_function="Logloss", auto_class_weights="Balanced",
                               random_seed=42, verbose=False)
        d.fit(Pool(df[feats_det].iloc[tr], fert_bin[tr]))
        det_p[te] = d.predict_proba(Pool(df[feats_det].iloc[te]))[:, list(d.classes_).index(1)]
        folds.append(int(g[te][0]))

    # blend the two Fertility evidences (fixed w), renormalize the other 3 classes
    blended = main_p.copy()
    new_fert = BLEND_W * main_p[:, FERT] + (1 - BLEND_W) * det_p
    rest = main_p.copy(); rest[:, FERT] = 0.0
    scale = (1 - new_fert) / np.maximum(rest.sum(1), 1e-9)
    blended = rest * scale[:, None]; blended[:, FERT] = new_fert

    # per-fold HSMM (durations from training subjects only -> leakage-free)
    def hsmm(P):
        out = np.zeros(len(P), int); le = np.log(np.clip(P, 1e-9, 1))
        for s in np.unique(g):
            te = g == s
            mu, sd = learn_durations(yi[~te], g[~te], days[~te])
            idx = np.where(te)[0]; order = idx[np.argsort(days[idx])]
            out[order] = seg_viterbi(le[order], mu, sd)
        return out

    ref = hsmm(blended)
    for tag, pred in [("single-stage (raw)", main_p.argmax(1)),
                      ("single-stage + HSMM", hsmm(main_p)),
                      ("two-stage + HSMM (REFERENCE)", ref)]:
        per = " ".join(f"{CLASSES[k][:4]}={f1_score(yi==k,pred==k):.3f}" for k in range(len(CLASSES)))
        print(f"  {tag:30} macroF1={f1_score(yi,pred,average='macro'):.3f} "
              f"acc={accuracy_score(yi,pred):.3f} | {per}")

    (BENCH / "splits").mkdir(parents=True, exist_ok=True)
    (BENCH / "results").mkdir(parents=True, exist_ok=True)
    json.dump({"scheme": "leave-one-subject-out", "n_folds": len(folds),
               "test_subject_per_fold": folds}, open(BENCH / "splits/loso_folds.json", "w"), indent=2)
    oof = df[["id", "day_in_study", "phase"]].copy()
    oof["pred"] = np.array(CLASSES)[ref]
    oof["pred_single_stage"] = np.array(CLASSES)[hsmm(main_p)]
    for k, c in enumerate(CLASSES):
        oof[f"proba_{c}"] = blended[:, k]
    oof.to_parquet(BENCH / "results/oof_reference.parquet", index=False)
    print("\nsaved splits -> benchmark/splits/loso_folds.json")
    print("saved OOF    -> benchmark/results/oof_reference.parquet")


if __name__ == "__main__":
    main()

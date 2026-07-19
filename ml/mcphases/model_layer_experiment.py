"""Model-layer experiment on the TEAM's canonical data (master_2022), reusing THEIR
feature construction (attempt1_ovulation base + ovulation anchor). Swaps only the
model (CatBoost vs TabPFN v2) and adds the HSMM decoder with data-LEARNED durations.
Standalone: does not edit any shared pipeline file. LOSO, warmup dropped.
"""
import warnings; warnings.filterwarnings("ignore")
import pathlib, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, accuracy_score
from catboost import CatBoostClassifier
sys.path.insert(0, ".")
from ai.hormonal.smoothing import hsmm_smooth

PROC = pathlib.Path("data/processed")
CLASSES = ["Menstrual", "Follicular", "Fertility", "Luteal"]
SYMPTOMS = ["flow_volume", "cramps", "bloating", "moodswing", "sorebreasts", "fatigue",
            "headaches", "appetite", "stress"]
WEARABLE = ["rhr_value", "sleep_resting_heart_rate", "hr_bpm_mean", "hrv_rmssd_mean",
            "resp_full_sleep_breathing_rate", "temp_nightly_temperature",
            "wtemp_temperature_diff_from_baseline", "stress_stress_score",
            "sleep_overall_score", "cgm_glucose_value_mean"]
GEN = {"Not at all": 0, "Very Low": 1, "Very Low/Little": 1, "Low": 2, "Moderate": 3,
       "High": 4, "Very High": 5, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
FV = {"Not at all": 0, "Spotting / Very Light": 1, "Light": 2, "Somewhat Light": 3,
      "Moderate": 4, "Somewhat Heavy": 5, "Heavy": 6, "Very Heavy": 7}


def base_features(df):
    df = df.sort_values(["id", "day_in_study"]).reset_index(drop=True)
    for c in SYMPTOMS:
        m = FV if c == "flow_volume" else GEN
        df[c] = pd.to_numeric(df[c].astype("object").map(m), errors="coerce")
    feats = list(SYMPTOMS) + [c for c in WEARABLE if c in df.columns]
    g = df.groupby("id", sort=False)
    new = []
    for c in feats:
        for w in (3, 5, 7):
            df[f"{c}_rm{w}"] = g[c].transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
            df[f"{c}_rs{w}"] = g[c].transform(lambda s, w=w: s.rolling(w, min_periods=1).std().fillna(0))
            new += [f"{c}_rm{w}", f"{c}_rs{w}"]
    df["days_since_flow"] = df["days_since_flow"].astype(float)
    return df, feats + new + ["days_since_flow"]


def anchor_features(df):
    g = df.groupby("id", sort=False); cols = []
    for c in ["temp_nightly_temperature", "rhr_value", "sleep_resting_heart_rate",
              "wtemp_temperature_diff_from_baseline"]:
        if c not in df.columns:
            continue
        base = g[c].transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
        df[f"{c}_anchor_dev"] = df[c] - base
        df[f"{c}_anchor_shift3"] = df[c] - g[c].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        df[f"{c}_anchor_slope"] = g[c].transform(lambda s: s.diff())
        cols += [f"{c}_anchor_dev", f"{c}_anchor_shift3", f"{c}_anchor_slope"]
    return df, cols


def learn_hsmm_params(df):
    """Population duration mean/sd per phase from the label runs (a prior, not a
    per-row feature — standard HSMM setup)."""
    durs = {k: [] for k in range(4)}
    for _, idx in df.groupby("id", sort=False).groups.items():
        seq = [CLASSES.index(p) for p in df.loc[idx, "phase"]]
        i = 0
        while i < len(seq):
            j = i
            while j + 1 < len(seq) and seq[j + 1] == seq[i]:
                j += 1
            durs[seq[i]].append(j - i + 1); i = j + 1
    mean = np.array([np.mean(durs[k]) if durs[k] else 5.0 for k in range(4)])
    sd = np.array([max(1.0, np.std(durs[k])) if durs[k] else 2.0 for k in range(4)])
    return mean, sd, {CLASSES[k]: (round(mean[k], 1), round(sd[k], 1), len(durs[k])) for k in range(4)}


def make_cat():
    return CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=6,
                              loss_function="MultiClass", auto_class_weights="Balanced",
                              random_seed=42, verbose=False)


def make_tabpfn():
    from tabpfn import TabPFNClassifier
    dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    return TabPFNClassifier(device=dev, n_estimators=4, ignore_pretraining_limits=True, random_state=42)


def loso_proba(df, feats, make):
    X = df[feats]; yi = np.array([CLASSES.index(p) for p in df["phase"]]); gr = df["id"].to_numpy()
    oofp = np.zeros((len(df), 4))
    for tr, te in LeaveOneGroupOut().split(X, yi, gr):
        m = make(); m.fit(X.iloc[tr], yi[tr]); oofp[te] = m.predict_proba(X.iloc[te])
    return oofp, yi


def show(tag, yi, pred):
    per = {CLASSES[k]: f1_score(yi == k, pred == k) for k in range(4)}
    print(f"  {tag:24s} macroF1={f1_score(yi, pred, average='macro'):.3f} "
          f"acc={accuracy_score(yi, pred):.3f} | Fert={per['Fertility']:.3f} "
          f"Mens={per['Menstrual']:.3f} Foll={per['Follicular']:.3f} Lute={per['Luteal']:.3f}")


def main():
    df = pd.read_parquet(PROC / "mcphases_master_2022.parquet")
    df = df[~df["is_warmup"] & df["phase"].isin(CLASSES)].copy()
    df, base = base_features(df); df, anch = anchor_features(df)
    feats = base + anch
    seg = df["id"].to_numpy()
    mean, sd, dtab = learn_hsmm_params(df)
    print(f"rows={len(df)} subjects={df.id.nunique()} feats={len(feats)}")
    print(f"learned durations (mean, sd, n): {dtab}\n")

    for name, make in [("CatBoost", make_cat), ("TabPFN v2", make_tabpfn)]:
        t = time.time()
        try:
            oofp, yi = loso_proba(df, feats, make)
        except Exception as e:
            print(f"[{name}] FAILED: {str(e).splitlines()[0][:90]}\n"); continue
        print(f"[{name}]  ({time.time()-t:.0f}s)")
        show("raw", yi, oofp.argmax(1))
        show("+HSMM(learned dur)", yi, hsmm_smooth(oofp, seg, mean_dur=mean, sd_dur=sd, skip_prob=0.02))
        print()


if __name__ == "__main__":
    main()

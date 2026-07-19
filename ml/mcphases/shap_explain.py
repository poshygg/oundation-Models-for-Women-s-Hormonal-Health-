"""Global SHAP feature importance for the B1 multimodal model.

Trains one CatBoost on all labeled days and uses CatBoost's exact TreeSHAP to rank
features. Sanity check: physiology-grounded features (temperature, HR, symptoms)
should dominate, confirming the model learned biology rather than an artifact.
"""
from __future__ import annotations
import pathlib, warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train import build_features, CLASSES, PROC

warnings.filterwarnings("ignore")


def main(baseline="B1", topn=20):
    df = pd.read_parquet(PROC / "mcphases_daily.parquet")
    df = df[df["phase"].isin(CLASSES)].reset_index(drop=True)
    df, feats, _ = build_features(df, baseline)
    X, y = df[feats], df["phase"]

    model = CatBoostClassifier(iterations=400, depth=5, learning_rate=0.05,
                               loss_function="MultiClass", auto_class_weights="Balanced",
                               random_seed=42, verbose=False)
    model.fit(Pool(X, y))

    pool = Pool(X, y)
    shap = model.get_feature_importance(pool, type="ShapValues")  # (n, classes, feats+1)
    # mean |SHAP| across samples and classes
    imp = np.abs(shap[:, :, :-1]).mean(axis=(0, 1))
    rank = pd.Series(imp, index=feats).sort_values(ascending=False)

    print(f"[{baseline}] Global SHAP importance (mean |value|), top {topn}:")
    for i, (f, v) in enumerate(rank.head(topn).items(), 1):
        print(f"  {i:2d}. {f:38} {v:.4f}")
    rank.to_csv(PROC / f"shap_importance_{baseline}.csv", header=["mean_abs_shap"])
    print(f"\nsaved -> {PROC / f'shap_importance_{baseline}.csv'}")


if __name__ == "__main__":
    main()

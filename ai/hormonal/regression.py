"""ai/hormonal/regression.py — hormone-level regression branch (LOSO).

The roadmap's second prong: predict continuous relative levels of the three Mira
urinary metabolites — LH, E3G (estrogen), PdG (progesterone) — from the same
multimodal daily features, under the identical leave-one-subject-out split as the
classifier so the two branches are directly comparable.

One regressor per hormone (LightGBM by default; falls back to sklearn's
HistGradientBoosting if LightGBM is unavailable). Metrics per hormone: MAE, RMSE,
R², and Spearman rho (trend agreement — what matters clinically for a hormone
curve), reported overall and split by cycle regularity.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut


def make_regressor(backend: str, params: dict):
    if backend == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**params, verbose=-1)
    if backend == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(**params)
    if backend == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(**params)
    raise ValueError(f"unknown regression backend: {backend}")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def run_loso_regression(ds, backend: str, params: dict, max_folds: int | None = None) -> dict:
    """LOSO out-of-fold hormone predictions + per-hormone metrics.

    Trains one regressor per hormone on every fold. Returns OOF predictions
    (n, n_hormones) and a metrics report."""
    if ds.y_reg is None:
        raise ValueError("dataset has no hormone targets (y_reg is None)")

    X, Y, groups = ds.X, ds.y_reg, ds.groups
    hormones = ds.hormone_names
    subjects = np.unique(groups)
    if max_folds:
        subjects = subjects[: int(max_folds)]

    oof = np.full_like(Y, np.nan, dtype=float)
    t0 = time.perf_counter()

    for test_subj in subjects:
        test_mask = groups == test_subj
        train_mask = ~test_mask
        if max_folds:
            train_mask &= np.isin(groups, subjects)
        for h in range(Y.shape[1]):
            ## Sparse assays (e.g. PdG): train only on rows where THIS hormone
            ## is measured. Predict all test rows; scoring masks missing truth.
            fit_mask = train_mask & ~np.isnan(Y[:, h])
            if fit_mask.sum() < 10:
                continue
            reg = make_regressor(backend, params)
            reg.fit(X[fit_mask], Y[fit_mask, h])
            oof[test_mask, h] = reg.predict(X[test_mask])

    total = time.perf_counter() - t0
    report = _regression_report(Y, oof, hormones, ds.regular)
    report["timing"] = {"total_seconds": round(total, 2), "n_folds": len(subjects),
                        "models_per_fold": Y.shape[1]}
    return {"oof": oof, "report": report, "backend": backend}


def _per_hormone(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ## Score only rows where both truth and prediction are present.
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if m.sum() < 3:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"),
                "spearman": float("nan"), "n": int(m.sum())}
    yt, yp = y_true[m], y_pred[m]
    rho = spearmanr(yt, yp).statistic if len(np.unique(yt)) > 1 else float("nan")
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": _rmse(yt, yp),
        "r2": float(r2_score(yt, yp)),
        "spearman": float(rho),
        "n": int(m.sum()),
    }


def _regression_report(Y: np.ndarray, P: np.ndarray, hormones: list[str],
                       regular: np.ndarray) -> dict:
    per = {h: _per_hormone(Y[:, i], P[:, i]) for i, h in enumerate(hormones)}
    valid = [h for h in hormones if not np.isnan(per[h]["r2"])]
    macro = {
        "mae": float(np.mean([per[h]["mae"] for h in valid])) if valid else float("nan"),
        "r2": float(np.mean([per[h]["r2"] for h in valid])) if valid else float("nan"),
        "spearman": float(np.nanmean([per[h]["spearman"] for h in valid])) if valid else float("nan"),
    }
    by_reg = {}
    for name, mask in (("regular", regular), ("irregular", ~regular)):
        if mask.sum() == 0:
            continue
        by_reg[name] = {h: _per_hormone(Y[mask, i], P[mask, i]) for i, h in enumerate(hormones)}
    return {"per_hormone": per, "macro": macro, "by_regularity": by_reg}

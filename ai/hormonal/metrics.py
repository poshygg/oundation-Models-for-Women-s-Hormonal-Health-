"""ai/hormonal/metrics.py — evaluation, no-call abstention, calibration.

The challenge rewards honest uncertainty over forced answers, so metrics are
reported three ways: overall, selectively (after no-call abstention), and split
by cycle regularity — the regime where prior models quietly degrade.

Primary comparison point: macro-F1 vs the mcPHASES SOTA (CatBoost+HSMM, LOSO,
0.662 macro-F1 / 67.6% acc, self-report symptoms only).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from .data import PHASES


def _onehot(y: np.ndarray, n_classes: int) -> np.ndarray:
    oh = np.zeros((len(y), n_classes), dtype=float)
    oh[np.arange(len(y)), y] = 1.0
    return oh


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean per-class Brier score (lower is better) — a calibration-sensitive
    metric averaged one-vs-rest across the 4 phases."""
    n_classes = proba.shape[1]
    oh = _onehot(y_true, n_classes)
    scores = [brier_score_loss(oh[:, k], proba[:, k]) for k in range(n_classes)]
    return float(np.mean(scores))


def safe_auroc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """OvR macro AUROC, guarding against folds where a class is absent."""
    present = np.unique(y_true)
    if len(present) < 2:
        return float("nan")
    try:
        return float(
            roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=np.arange(proba.shape[1]))
        )
    except ValueError:
        ## Fall back to only-present classes if a full-label AUROC is undefined.
        return float(roc_auc_score(_onehot(y_true, proba.shape[1])[:, present], proba[:, present], average="macro"))


def core_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    y_pred = proba.argmax(axis=1)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=np.arange(proba.shape[1]))),
        "auroc_ovr": safe_auroc(y_true, proba),
        "brier": multiclass_brier(y_true, proba),
    }


def apply_nocall(proba: np.ndarray, min_confidence: float) -> np.ndarray:
    """Boolean mask of rows the model is confident enough to answer."""
    return proba.max(axis=1) >= min_confidence


def selective_metrics(y_true: np.ndarray, proba: np.ndarray, min_confidence: float) -> dict:
    """Metrics computed only on answered rows, plus coverage (answer rate).
    A model that abstains on hard cases should show higher selective accuracy at
    coverage < 1.0 — the honest way to trade coverage for reliability."""
    mask = apply_nocall(proba, min_confidence)
    coverage = float(mask.mean())
    if mask.sum() == 0:
        return {"coverage": 0.0, "selective_accuracy": float("nan"), "selective_macro_f1": float("nan")}
    m = core_metrics(y_true[mask], proba[mask])
    return {
        "coverage": coverage,
        "selective_accuracy": m["accuracy"],
        "selective_macro_f1": m["macro_f1"],
    }


def metrics_by_regularity(y_true: np.ndarray, proba: np.ndarray, regular: np.ndarray) -> dict:
    """Report regular vs irregular cycles separately — irregular is where prior
    work degrades from ~85% to ~50–80% and glosses over it."""
    out = {}
    for name, mask in (("regular", regular), ("irregular", ~regular)):
        if mask.sum() == 0:
            continue
        m = core_metrics(y_true[mask], proba[mask])
        out[name] = {"n": m["n"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "balanced_accuracy": m["balanced_accuracy"]}
    return out


def full_report(y_true: np.ndarray, proba: np.ndarray, regular: np.ndarray,
                min_confidence: float, nocall_enabled: bool) -> dict:
    rep = {"overall": core_metrics(y_true, proba)}
    if nocall_enabled:
        rep["selective"] = selective_metrics(y_true, proba, min_confidence)
    rep["by_regularity"] = metrics_by_regularity(y_true, proba, regular)
    rep["per_class_f1"] = {
        PHASES[k]: float(f)
        for k, f in enumerate(
            f1_score(y_true, proba.argmax(axis=1), average=None, labels=np.arange(proba.shape[1]))
        )
    }
    return rep

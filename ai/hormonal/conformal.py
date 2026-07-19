"""ai/hormonal/conformal.py — split-conformal prediction sets for the phase task.

Upgrades the ad-hoc confidence-threshold no-call into a distribution-free
prediction set with a finite-sample marginal coverage guarantee: at level
``alpha`` the returned set contains the true phase with probability >= 1 - alpha,
regardless of the underlying model (CatBoost / TabPFN / EBM) or data distribution.

Two nonconformity scores are provided:
  * LAC  (least-ambiguous set-valued classifier): score = 1 - p(true). Smallest
    sets, marginal coverage.
  * APS  (adaptive prediction sets): cumulative probability mass down to the true
    label. Better class-conditional coverage; the default.

It runs on the out-of-fold probabilities the LOSO loop already produced, so it
costs no extra model fits. To keep calibration and test disjoint at the SUBJECT
level under LOSO, the conformal layer uses its own subject-grouped K-fold: qhat
is estimated on other subjects' OOF scores and applied to the held-out subjects',
then pooled over all subjects.

No-call mapping: a prediction is decisive when its set is a singleton; a set with
0 or >1 phases is an abstention (no-call) — honest, and now with a guarantee.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold


def _qhat(scores: np.ndarray, alpha: float) -> float:
    """Conformal quantile with the finite-sample correction ceil((n+1)(1-a))/n."""
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


## ----------------------------------------------------------------------------
## APS (adaptive prediction sets)
## ----------------------------------------------------------------------------
def _aps_cal_scores(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    order = np.argsort(-proba, axis=1)
    sorted_p = np.take_along_axis(proba, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    ranks = (order == y[:, None]).argmax(axis=1)   # position of true label in desc order
    return cum[np.arange(len(y)), ranks]


def _aps_predict_mask(proba: np.ndarray, qhat: float) -> np.ndarray:
    order = np.argsort(-proba, axis=1)
    sorted_p = np.take_along_axis(proba, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    prev = cum - sorted_p                          # cumulative BEFORE this class
    include_sorted = prev < qhat                   # include up to and incl. the crossing class
    mask = np.zeros_like(proba, dtype=bool)
    np.put_along_axis(mask, order, include_sorted, axis=1)
    return mask


## ----------------------------------------------------------------------------
## LAC (least-ambiguous set-valued classifier)
## ----------------------------------------------------------------------------
def _lac_cal_scores(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 1.0 - proba[np.arange(len(y)), y]


def _lac_predict_mask(proba: np.ndarray, qhat: float) -> np.ndarray:
    return proba >= (1.0 - qhat)


## ----------------------------------------------------------------------------
## OOF conformal (subject-grouped, no refits)
## ----------------------------------------------------------------------------
def conformal_oof(proba: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  alpha: float, method: str = "aps", n_splits: int = 5) -> tuple[np.ndarray, list[float]]:
    """Return (set_mask, per-fold qhats). ``set_mask[i, c]`` is True when phase c
    is in the conformal set for row i."""
    n, K = proba.shape
    mask = np.zeros((n, K), dtype=bool)
    n_splits = int(min(n_splits, len(np.unique(groups))))
    gkf = GroupKFold(n_splits=n_splits)
    qhats: list[float] = []
    for cal_idx, test_idx in gkf.split(proba, y, groups):
        ## GroupKFold's larger "train" split is the calibration set; the smaller
        ## fold receives prediction sets. All subject-disjoint.
        if method == "aps":
            scores = _aps_cal_scores(proba[cal_idx], y[cal_idx])
            qh = _qhat(scores, alpha)
            mask[test_idx] = _aps_predict_mask(proba[test_idx], qh)
        elif method == "lac":
            scores = _lac_cal_scores(proba[cal_idx], y[cal_idx])
            qh = _qhat(scores, alpha)
            mask[test_idx] = _lac_predict_mask(proba[test_idx], qh)
        else:
            raise ValueError(f"unknown conformal method: {method}")
        qhats.append(qh)
    return mask, qhats


def conformal_report(mask: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    n = len(y)
    sizes = mask.sum(axis=1)
    covered = mask[np.arange(n), y]
    singleton = sizes == 1
    empty = sizes == 0
    if singleton.any():
        sing_pred = mask[singleton].argmax(axis=1)
        sing_acc = float((sing_pred == y[singleton]).mean())
    else:
        sing_acc = float("nan")
    return {
        "method_alpha": alpha,
        "target_coverage": round(1 - alpha, 4),
        "empirical_coverage": float(covered.mean()),      # should be >= target
        "avg_set_size": float(sizes.mean()),
        "singleton_rate": float(singleton.mean()),         # decisive predictions
        "singleton_accuracy": sing_acc,                    # accuracy when decisive
        "empty_rate": float(empty.mean()),
        "abstain_rate": float((sizes != 1).mean()),        # no-call = non-singleton
        "set_size_hist": {int(k): int((sizes == k).sum()) for k in range(mask.shape[1] + 1)},
    }

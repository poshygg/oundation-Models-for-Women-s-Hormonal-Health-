"""ai/hormonal/smoothing.py — biological temporal smoothing of phase predictions.

A per-day classifier has no notion that menstrual phases run in a fixed cyclic
order (Menstrual -> Follicular -> Fertility -> Luteal -> Menstrual) with typical
durations, so it flip-flops day to day. This decodes the classifier's per-day
probabilities through that biology and returns a temporally coherent sequence —
the piece SOTA (CatBoost + HSMM) adds on top of its classifier.

Two decoders:
  * ``hsmm_smooth``  — Hidden SEMI-Markov: models each phase's DURATION explicitly
    (Gaussian prior), decoded by segmental Viterbi. Because the cyclic order is
    enforced and Fertility carries a tight ~2-day duration prior, a short Fertility
    segment is inserted even when the per-day signal is weak — which is exactly what
    an ordinary HMM erases. This is the version that can lift macro-F1.
  * ``hmm_smooth``   — geometric-duration HMM (self/advance transitions). Cheaper,
    lifts accuracy but tends to skip the 2-day Fertility window. Kept for reference.

Default entry point ``smooth_proba`` uses the HSMM.
"""

from __future__ import annotations

import numpy as np

from .data import PHASE_DURATIONS, PHASES

K = len(PHASES)
_MEAN_DUR = np.array([PHASE_DURATIONS[p] for p in PHASES], dtype=float)
## Duration spreads: Fertility deliberately tight so it stays a short window and is
## neither skipped nor stretched; longer phases get more slack.
_SD_DUR = np.array([2.0, 3.0, 1.0, 4.0], dtype=float)
_NEG = -1e18


## ----------------------------------------------------------------------------
## Segment-to-segment transition (cyclic advance, small skip mass)
## ----------------------------------------------------------------------------
def _seg_log_trans(skip_prob: float) -> np.ndarray:
    t = np.full((K, K), _NEG)
    for i in range(K):
        t[i, (i + 1) % K] = np.log(max(1e-9, 1.0 - skip_prob))   # advance one phase
        t[i, (i + 2) % K] = np.log(max(1e-9, skip_prob))          # skip (e.g. anovulatory)
    return t


def _log_dur(durations: int, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    ## Discretized Gaussian log-density over duration d = 1..durations, per phase.
    d = np.arange(1, durations + 1)[:, None]                       # (D, 1)
    return -0.5 * ((d - mean[None, :]) / sd[None, :]) ** 2 - np.log(sd[None, :])  # (D, K)


## ----------------------------------------------------------------------------
## HSMM segmental Viterbi (explicit durations)
## ----------------------------------------------------------------------------
def _hsmm_viterbi(log_emit: np.ndarray, log_trans: np.ndarray, log_dur: np.ndarray,
                  d_max: int) -> np.ndarray:
    n, k = log_emit.shape
    ## Prefix sums of emissions for O(1) segment sums. cum[j, t] = sum_{s<t} emit[s,j].
    cum = np.zeros((k, n + 1))
    cum[:, 1:] = np.cumsum(log_emit.T, axis=1)
    log_start = np.log(1.0 / k)

    best = np.full((n, k), _NEG)
    back = np.empty((n, k, 2), dtype=int)   # (chosen duration, previous phase or -1)
    for t in range(n):
        dmax = min(d_max, t + 1)
        for j in range(k):
            for d in range(1, dmax + 1):
                a = t - d + 1
                seg = (cum[j, t + 1] - cum[j, a]) + log_dur[d - 1, j]
                if a == 0:                                   # first segment
                    sc = log_start + seg
                    if sc > best[t, j]:
                        best[t, j] = sc; back[t, j] = (d, -1)
                else:
                    prev = best[a - 1] + log_trans[:, j]     # (k,)
                    i = int(np.argmax(prev))
                    sc = prev[i] + seg
                    if sc > best[t, j]:
                        best[t, j] = sc; back[t, j] = (d, i)

    labels = np.empty(n, dtype=int)
    t, j = n - 1, int(np.argmax(best[n - 1]))
    while t >= 0:
        d, i = back[t, j]
        labels[t - d + 1:t + 1] = j
        t -= d
        if i < 0:
            break
        j = i
    return labels


def hsmm_smooth(proba: np.ndarray, segment: np.ndarray, skip_prob: float = 0.05,
                mean_dur: np.ndarray | None = None, sd_dur: np.ndarray | None = None,
                d_max: int = 45) -> np.ndarray:
    """HSMM (explicit-duration) smoothing. Returns a phase label per row."""
    mean_dur = _MEAN_DUR if mean_dur is None else np.asarray(mean_dur, float)
    sd_dur = _SD_DUR if sd_dur is None else np.asarray(sd_dur, float)
    log_trans = _seg_log_trans(skip_prob)
    log_dur = _log_dur(d_max, mean_dur, sd_dur)
    log_emit_all = np.log(np.clip(proba, 1e-9, None))
    out = np.empty(len(proba), dtype=int)
    for seg in np.unique(segment):
        idx = np.flatnonzero(segment == seg)               # already day-ordered
        out[idx] = _hsmm_viterbi(log_emit_all[idx], log_trans, log_dur, d_max)
    return out


## ----------------------------------------------------------------------------
## Geometric-duration HMM (reference; lifts accuracy, tends to drop Fertility)
## ----------------------------------------------------------------------------
def _hmm_log_trans(mean_dur: np.ndarray, skip_prob: float) -> np.ndarray:
    t = np.zeros((K, K))
    for i in range(K):
        stay = np.clip((mean_dur[i] - 1.0) / mean_dur[i], 1e-3, 1 - 1e-3)
        t[i, i] = stay
        t[i, (i + 1) % K] = 1.0 - stay - skip_prob
        t[i, (i + 2) % K] = skip_prob
    t = np.clip(t, 1e-9, None)
    return np.log(t / t.sum(axis=1, keepdims=True))


def hmm_smooth(proba: np.ndarray, segment: np.ndarray, skip_prob: float = 0.02) -> np.ndarray:
    log_trans = _hmm_log_trans(_MEAN_DUR, skip_prob)
    log_emit = np.log(np.clip(proba, 1e-9, None))
    out = np.empty(len(proba), dtype=int)
    for seg in np.unique(segment):
        idx = np.flatnonzero(segment == seg)
        e = log_emit[idx]
        n = len(idx)
        delta = np.full((n, K), -np.inf); back = np.zeros((n, K), dtype=int)
        delta[0] = np.log(1.0 / K) + e[0]
        for t in range(1, n):
            scored = delta[t - 1][:, None] + log_trans
            back[t] = np.argmax(scored, axis=0)
            delta[t] = scored[back[t], np.arange(K)] + e[t]
        path = np.empty(n, dtype=int); path[-1] = int(np.argmax(delta[-1]))
        for t in range(n - 2, -1, -1):
            path[t] = back[t + 1, path[t + 1]]
        out[idx] = path
    return out


def smooth_proba(proba: np.ndarray, segment: np.ndarray, method: str = "hsmm", **kw) -> np.ndarray:
    """Temporally coherent phase labels. method: 'hsmm' (default) or 'hmm'."""
    return hsmm_smooth(proba, segment, **kw) if method == "hsmm" else hmm_smooth(proba, segment, **kw)

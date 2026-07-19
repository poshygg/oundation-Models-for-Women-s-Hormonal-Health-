# Four-phase menstrual cycle classification on mcPHASES

A reproducible model and an open benchmark for inferring the four menstrual cycle
phases — menstrual, follicular, fertility (ovulatory), and luteal — from daily
self-reported symptoms and wearable physiology. Phase labels in mcPHASES are derived
from at-home hormone assays, so we treat the hormone readings (LH, estradiol, PdG) as
label sources and keep them out of the feature set; the model never sees them.

This is a Layer 02 (AI Model) submission that also leaves behind a Layer 01 artifact:
a frozen evaluation split and a model-agnostic scoring harness that anyone can point a
new model at.

## What problem this addresses

The challenge names three gaps. Our work targets all three directly rather than
building around them.

*No shared benchmark.* There is no standard, transparent way to compare four-phase
cycle models on this data — most work reports a single number on an undocumented split.
We ship one: a fixed leave-one-subject-out split (`benchmark/splits/loso_folds.json`),
a bootstrap confidence-interval harness that scores any set of out-of-fold predictions
(`benchmark/eval/ci_harness.py`), and a reference set of predictions to check against.
Drop in a model, run the harness, get a comparable interval.

*Fragmented infrastructure.* Everything needed to reproduce the result lives in one
repository — the data-assembly script, the model, the split, the evaluation, and a
model card — under an open license (the mcPHASES raw data itself stays out, per its
PhysioNet terms).

*Dynamic biology, static care.* The model reads a continuous daily stream of symptom
and wearable signals and produces a phase estimate for every day, rather than treating
the cycle as something you check once in a clinic.

## Task definition and scoring

We evaluate leave-one-subject-out over 42 participants: train on 41, test on the held-out
person, repeat. This is the honest split for this data — a random day-level split would
let the model memorize a participant's baseline symptom style instead of learning phase
signatures that transfer to a new person.

Because the cohort is small (N=42), a single point estimate is misleading. We report
macro-F1 and accuracy with 95% confidence intervals from a subject-level bootstrap
(B=2000 resamples). Everything below is out-of-fold; no hyperparameter is tuned on the
reported test split.

## Data preparation

mcPHASES spans two observation windows. The 2024 window has symptoms recorded on well
under 1% of days, so we restrict to the 2022 window. Within it we drop a 14-day warm-up
per participant (needed before rolling statistics stabilize) and remove runs of five or
more consecutive days with no self-report, which would otherwise inject imputation
artifacts into the rolling features. That leaves 3,110 participant-days across 42 people
for the multimodal view (`build_master.py`).

Reconciling the data this way was, on its own, the single largest change in the whole
project — it moved macro-F1 by roughly +0.09 before any modeling, mostly by removing
empty symptom rows.

## Model

The pipeline is a two-stage predictor with temporal smoothing on top.

**Features.** Four groups, all computed within each participant and using past days only:

- cycle position from self-reported flow (a days-since-last-bleed counter with several
  onset thresholds, so spotting and long periods don't reset it spuriously);
- rolling means and standard deviations of each symptom over 3–14 day windows;
- per-subject baseline deviations of the wearable channels (nightly skin temperature,
  resting heart rate, HRV), which carry the luteal-phase shifts;
- ovulation anchors built from the temperature and heart-rate rise around mid-cycle.

The cycle-position counter is derived from reported flow, never from the phase label.
We flag this because an earlier version counted from the label itself and inflated the
score by about 0.17 — a reminder that a dominant feature always deserves an ablation.

**Stage 1 — per-day classification.** A gradient-boosted classifier produces a four-class
probability for each day. The backend is interchangeable (see Results); we default to
CatBoost for its native missing-value handling, which matters when ~40% of symptom
entries are absent.

**Stage 1b — a dedicated fertility detector.** Fertility is the hardest phase for every
approach: it lasts only a couple of days and its symptom shift is subtle. Instead of
hoping the four-class model handles it, we train a separate binary "fertility vs. rest"
detector on ovulation-anchored temperature, heart-rate, and cycle-position features. Its
out-of-fold fertility probability is blended with the main model's at a fixed 0.5 weight
— fixed in advance, so there is no selection on the test split.

**Stage 2 — HSMM smoothing.** A hidden semi-Markov model enforces the two things a
per-day classifier cannot: the cyclic phase order (menstrual → follicular → fertility →
luteal) and realistic phase durations. Durations are learned as Gaussian priors from the
training participants of each fold, so nothing leaks from the held-out person.

**Abstention.** Split conformal prediction wraps the probabilities and returns a
prediction set with a distribution-free coverage guarantee. A single-phase set is a
confident call; an empty or multi-phase set is a no-call. This is what lets the model
say "unsure" on an irregular cycle instead of forcing a wrong phase.

**Explanation.** TreeSHAP gives global and per-phase feature attributions, which line up
with known physiology — flow-derived cycle position and luteal temperature/heart-rate
shifts carry most of the signal.

## Results

All numbers are out-of-fold, LOSO over 42 participants.

### Backend choice is not the lever

Swapping the Stage-1 classifier under one identical split and feature matrix (single-stage,
four-class, with HSMM smoothing):

| backend | macro-F1 | accuracy | menstrual | follicular | fertility | luteal |
|---|---|---|---|---|---|---|
| XGBoost | 0.661 | 0.679 | 0.740 | 0.658 | 0.491 | 0.757 |
| CatBoost | 0.651 | 0.668 | 0.742 | 0.617 | 0.482 | 0.762 |
| LightGBM | 0.651 | 0.670 | 0.733 | 0.649 | 0.471 | 0.753 |

The three tree backends land within about 0.01 of each other — indistinguishable at this
sample size. XGBoost edges ahead here, mostly on the follicular phase. The takeaway is
that the classifier family is not where the score comes from; the features and the
temporal model are.

### The two-stage model

Adding the fertility detector on top of the base classifier:

| model | macro-F1 [95% CI] | accuracy | fertility F1 |
|---|---|---|---|
| base classifier + HSMM | 0.654 [0.601, 0.701] | 0.671 | 0.498 |
| two-stage (+ fertility detector) | **0.667 [0.620, 0.709]** | 0.678 | **0.531** |

The detector lifts fertility F1 from 0.498 to 0.531 — the phase that matters most for any
fertility-tracking use and the one every method struggles with — while nudging the
overall macro-F1 up a little. The confidence interval is what we report; a bare
point estimate at N=42 would overstate precision.

### Abstention behavior

Conformal coverage lands at the 0.90 target across backends (empirical 0.898–0.900). When
the model is decisive (a single-phase set) its accuracy is 0.85–0.89; the rest become
no-calls rather than confident errors.

## What ships

The `benchmark/` directory is the reusable part:

- `splits/loso_folds.json` — the frozen 42-fold subject split;
- `eval/ci_harness.py` — model-agnostic scorer; feed it any out-of-fold prediction table
  and it returns macro-F1 / accuracy with subject-bootstrap CIs;
- `eval/reference_model.py` — the two-stage model above, runnable end to end;
- `results/oof_reference.parquet` — reference predictions to compare against;
- `MODEL_CARD.md`, `README.md`, `LICENSE` (MIT; raw mcPHASES excluded).

The intent is that a future team can replace the model, keep the split and scorer, and
report a number that is directly comparable to ours.

## Reproducing

From the repository root:

```
python ml/mcphases/build_master.py            # assemble the aligned daily table
python benchmark/eval/reference_model.py       # train + write OOF predictions
python benchmark/eval/ci_harness.py            # score with bootstrap CIs
python ml/mcphases/backend_compare_honest.py   # CatBoost / XGBoost / LightGBM comparison
```

Seeds are fixed (42) and the split is frozen, so the numbers above regenerate.

## Limitations

The cohort is 42 young adults recruited in one region, predominantly of East/Southeast
Asian and Caucasian background, with no BMI recorded and hormonal contraception excluded
— external validity to older, more diverse, or clinically complex populations is untested.
Phase labels come from hormone assays but phase boundaries are biologically gradual, which
puts a ceiling on achievable accuracy near transitions. Fertility remains the weakest
phase despite the dedicated detector. The model leans on reported flow for cycle position,
so it degrades for users who log flow inconsistently.

This is research and infrastructure, not a medical device, and makes no diagnostic claim.
When confidence is low the model abstains rather than assigning a phase.

## Reference point

The starting point for the modeling choices — a CatBoost classifier with an HSMM temporal
layer over self-reported symptoms, evaluated leave-one-subject-out — follows the
established approach for this dataset (medRxiv, 2026). Our additions are the dedicated
fertility detector, the conformal abstention layer, and the packaged benchmark.

# Model Card — mcPHASES 4-Phase Cycle Classifier (reference)

Reference model for the mcPHASES 4-phase benchmark. Optimized for reproducibility,
scientific rigor, and explainability over size — a reusable building block, not a
foundation model.

## Model details

- **Type:** two-stage. A 4-class gradient-boosted tree model (CatBoost) + a dedicated
  binary **Fertility (ovulation-window) detector** whose out-of-fold probability is blended
  (fixed 0.5 weight) into the 4-class Fertility probability, then an explicit-duration
  Hidden Semi-Markov Model (HSMM) temporal decoder. The detector targets the ovulation
  window — the hardest class for every method — with the biphasic temperature/HR shift.
- **Task:** 4-phase menstrual-cycle classification (Menstrual / Follicular / Fertility /
  Luteal), one prediction per subject-day.
- **Inputs:** multimodal daily features from mcPHASES — self-report symptoms (ordinal),
  Fitbit wearables (resting HR, HR, HRV, nightly skin temperature, wrist-temp deviation,
  respiratory rate, sleep, stress, activity, VO₂max), Dexcom CGM (glucose mean/std), plus
  physiology-grounded engineering: 3/5/7-day rolling mean+std, day-over-day deltas,
  causal ovulation-shift anchors (temperature/HR), and a **leakage-free onset-based
  cycle-position** feature derived from self-reported flow.
- **Excluded from inputs:** hormone measurements (LH / E3G / PdG) — they define the
  labels, so using them would be target leakage.
- **Decoder:** HSMM with cyclic phase order and Gaussian duration priors **learned per
  fold** (training subjects only) + a small skip probability for anovulatory cycles.

## Training & evaluation

- **Data:** mcPHASES 2022 interval, multimodal view — 3,110 subject-days, 42 subjects
  (see `README.md` for the alignment/filtering rationale).
- **Protocol:** leave-one-subject-out (42 folds). All feature engineering is within-subject
  and past-only; HSMM durations are fit on training subjects only → no cross-subject or
  target leakage.
- **No selection on the test folds:** feature set and decoder settings are fixed a priori;
  no ensemble-weight or feature-count sweep is tuned on the reported split.
- **Hyperparameters:** CatBoost 500 iterations, depth 6, lr 0.05, L2 leaf reg 6, balanced
  class weights, seed 42.

## Metrics (subject-bootstrap 95% CI, B=2000)

| Metric | Score | 95% CI |
|---|---|---|
| macro-F1 | 0.667 | [0.620, 0.709] |
| accuracy | 0.678 | [0.631, 0.721] |
| F1 Menstrual | 0.747 | [0.692, 0.797] |
| F1 Follicular | 0.627 | [0.576, 0.674] |
| F1 Fertility | 0.531 | [0.464, 0.595] |
| F1 Luteal | 0.761 | [0.714, 0.803] |

Ablation without the Fertility detector (single-stage + HSMM): macro-F1 0.654, Fertility
F1 0.498. The detector lifts Fertility +0.033 and macro-F1 +0.013 with a fixed blend
weight (no selection on the test split).

Published SOTA (symptom-only CatBoost+HSMM): macro-F1 0.662 / acc 0.676. The reference
model's CI contains SOTA with the point estimate slightly above → **statistically on
par**, achieved leakage-free.

## Explainability & uncertainty

- **SHAP** (exact TreeSHAP) attributes each prediction; top drivers are cycle position and
  physiologically expected signals (menstrual flow, resting-HR / respiratory-rate / skin-
  temperature deviations). Used to *audit* the model (confirm biology, not artifacts).
- **Conformal no-call** (model-agnostic, LAC/APS): distribution-free prediction sets at
  ~90% coverage; the model abstains ("no-call") when the set is ambiguous instead of
  guessing — honest behavior on irregular cycles.

## Intended use

Research building block for wearable-based hormonal-state prediction: a reproducible
baseline others can extend, and a harness to compare methods on identical splits. **Not a
medical or diagnostic device.**

## Limitations & risks

- **Small, narrow cohort** (N=42, ages 18–29, limited demographics, no hormonal-therapy
  users) → limited external validity.
- **Fertility is the hardest class** (~0.50 F1) for all methods; the ovulation window is
  short (~3 days) with subtle signals. Do not use for contraception/conception decisions.
- **Flow dependence:** the cycle-position feature relies on self-reported menstruation;
  performance degrades for non-loggers. Wearable channels partially mitigate this.
- **Interval scope:** 2022 only; device firmware differed in 2024, and self-report there
  is ~99% missing.

## Reproducibility

Frozen splits (`splits/loso_folds.json`), reference OOF (`results/oof_reference.parquet`),
and one-command reproduction (`benchmark/README.md`). Code MIT-licensed; mcPHASES raw data
under PhysioNet terms (not redistributed).

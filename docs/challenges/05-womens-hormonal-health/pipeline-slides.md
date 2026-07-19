# mcPHASES Phase Classifier — Pipeline Documentation (slide-ready)

Challenge 05 / Layer 02. Bullet-point summary of preprocessing, feature selection, and
training for slide creation. Numbers are from the real mcPHASES LOSO run.

## 1. Dataset & Task
- **Data:** mcPHASES (PhysioNet) — 42 participants, 5,658 hormone-labeled subject-days, two 3-month intervals (2022 / 2024).
- **Modalities:** Fitbit Sense (HR, HRV, skin temp, respiration, sleep, stress, activity) + Dexcom G6 CGM + Mira urine hormones (LH / E3G / PdG) + daily symptom diary.
- **Primary task:** 4-phase classification — Menstrual / Follicular / Fertility / Luteal (hormone-verified labels).
- **Secondary task:** hormone-level regression (LH / E3G / PdG).

## 2. Preprocessing
- Merged the 23 raw mcPHASES tables into one **daily table**, keyed on `id` + `study_interval` + `day_in_study`.
- High-frequency signals (HR, HRV, glucose, steps) **aggregated to daily statistics** (mean / std / sum); nightly signals keyed on sleep-start day.
- **Symptoms ordinal-encoded** (Not at all < Very Low < Low < Moderate < High < Very High); `flow_volume` ordinal; `flow_color` nominal.
- **No imputation of raw values** — gradient-boosted trees handle missing values natively; per-segment median fill only for rolling-stat inputs.
- **Hormones (LH / estrogen / PdG) excluded from features** — they define the labels, so using them would be target leakage.

## 3. Feature Engineering (LOSO-safe)
- Computed **within each participant × interval only**, so no information crosses the train/test boundary.
- Per signal: **3- and 5-day rolling mean + standard deviation**, **day-over-day delta**, **deviation from personal baseline** (25th percentile).
- **`days_since_bleed`** — cycle-position feature, derived from reported menstrual flow (not from the label).
- Expansion: **~33 base signals → ~233 engineered features.**
- Emphasis on **within-person variability** (rolling std) and **personal baselines**, because inter-individual differences exceed phase-related changes (consistent with prior work: within-person variability was the dominant signal).

## 4. Feature Selection
- **Domain-curated base signals** chosen for physiological relevance — not every available column.
- **Ablation switch** (`symptoms_only` / `wearables_only` / `all`) to measure each group's contribution and reproduce the symptom-only benchmark.
- **No automatic pre-selection:** trees perform implicit selection (ignore irrelevant features); at N=42, explicit selection would overfit the selection itself.
- **Post-hoc validation with SHAP** — confirmed physiologically expected drivers dominate.
- Top signals by SHAP: `days_since_bleed`, menstrual flow, respiratory rate, resting HR, skin temperature, cramps, HRV.
- **Flagged for review:** `VO₂max` and activity minutes ranked high but lack a phase link (possible per-person fingerprint / confounder) — candidates to drop pending an ablation.

## 5. Training Setup
- **Split:** Leave-One-Subject-Out (LOSO), 42 folds — evaluates generalization to an unseen participant (the honest, hard split; prevents per-person memorization).
- **Model:** CatBoost, selected after a head-to-head of LightGBM / XGBoost / CatBoost / EBM / TabPFN under one identical LOSO split.
- **Hyperparameters:** 600 iterations, depth 6, learning rate 0.05, L2 leaf reg 6.0, balanced class weights.
- **Temporal post-processing:** Hidden Semi-Markov Model — cyclic phase order + learned Gaussian duration priors, decoded with segmental Viterbi (removes physiologically impossible day-to-day jumps).
- **Uncertainty:** split conformal prediction (target 90% coverage, class-conditional / Mondrian) → **no-call abstention** with a distribution-free coverage guarantee.
- **Explainability:** SHAP (tree backends) or EBM glassbox (intrinsic, no post-hoc pass).

## 6. Results (LOSO, real mcPHASES)
- **CatBoost: macro-F1 0.670 / accuracy 0.673** — matches the published SOTA (0.662 / 0.676).
- Backend ranking: CatBoost 0.670 > XGBoost 0.663 > LightGBM 0.662 > **TabPFN 0.644** > EBM 0.617.
- **TabPFN (tabular foundation model) did not beat the trees** — confirms boosted trees win on this small-N tabular data.
- **Conformal coverage ~0.90 across all backends** (model-agnostic guarantee holds).
- Regression: R² ≈ 0 (absolute hormone levels not recoverable from wearables); Spearman ≈ 0.21 (weak trend only).

## 7. Key Design Principles
- **No leakage:** within-subject feature engineering + LOSO + hormones excluded from inputs.
- **Relative over absolute:** normalize to each person's baseline; variability > raw level.
- **Honest ceiling:** all five models converge at ~0.66–0.67 → this is the data's limit, not a modeling failure; near-1.0 is not achievable without label leakage or an easier task.
- **Reproducible & open:** one shared LOSO split + feature matrix for all models; frozen splits + code.

# ai/hormonal — mcPHASES phase classifier + hormone regression (Challenge 05)

Layer-02 contribution for *Foundation Models for Women's Hormonal Health*: a
reproducible, explainable, **leave-one-subject-out** pipeline over multimodal
daily signals (Fitbit + CGM + self-report symptoms), with two branches:

- **Classification** — 4-phase cycle state (menses / late-follicular / ovulation / luteal).
- **Regression** — continuous relative levels of the Mira hormones LH / E3G / PdG.

Model choices, all swappable and directly compared (`compare.py`):
- **Gradient-boosted trees** — LightGBM / XGBoost / CatBoost (literature default for this tabular small-N data).
- **EBM** (Explainable Boosting Machine) — glassbox GAM; explanations are free (shape functions = the model, no SHAP pass).
- **TabPFN** — pre-trained transformer, in-context learning (no gradient training). Needs a one-time license token (see below).
- **Conformal prediction** — distribution-free no-call with a coverage guarantee, on top of any backend.

Prior-work basis: `docs/challenges/05-womens-hormonal-health/research-notes.md`.
The field's SOTA on mcPHASES is **CatBoost+HSMM, LOSO, macro-F1 0.662 / 67.6% acc**
using self-report symptoms *only*. Our opening is to add wearable + CGM features
on top of the symptom features under the same honest LOSO split.

## Quick start

```powershell
pip install -r requirements/ml_extra.txt      # catboost/xgboost/lightgbm/shap
# from the repo root:
python -m ai.hormonal.train --config ai/hormonal/configs/phase-clf-baseline.yaml
```

No mcPHASES download? It still runs. `data.py` synthesizes a physiology-grounded
dataset (luteal skin-temp +~0.3 °C, elevated resting HR, depressed HRV, phase-
specific symptoms) so the full pipeline is exercisable today. Drop the real
processed table at `data/processed/mcphases_features.parquet` and the same
command trains on real data — nothing else changes.

Fast smoke run (few folds, skip final SHAP):
```powershell
python -m ai.hormonal.train --config ai/hormonal/configs/phase-clf-baseline.yaml --max-folds 4 --no-final
```

Key ablation — reproduce the SOTA input, then show the wearable lift:
```powershell
python -m ai.hormonal.train --config ... --feature-set symptoms_only   # SOTA input
python -m ai.hormonal.train --config ... --feature-set all             # + Fitbit + CGM
```

Swap the model: `--backend xgboost` / `lightgbm` / `ebm` / `tabpfn` (CatBoost default in the baseline config).

## Compare everything (both branches, one LOSO split)

```powershell
python -m ai.hormonal.compare --config ai/hormonal/configs/compare.yaml
python -m ai.hormonal.compare --config ai/hormonal/configs/compare.yaml --max-folds 6   # fast
```

Runs each classification backend AND the hormone-regression branch on the same
data and split, then writes `experiments/hormonal/compare/comparison_report.md`
(macro-F1 / acc / timing / glassbox / conformal coverage per model, plus per-
hormone MAE/RMSE/R²/Spearman).

## Clinical report from model outputs (`explain.py`)

Turns the pipeline's explanation + prediction artifacts into a rigorous,
per-participant clinical-context report (roadmap Block 3). It reads the **real**
processed artifacts directly:

```powershell
python -m ai.hormonal.explain --subject 1
python -m ai.hormonal.explain --subject 1 --interval 2022 --provider openai
python -m ai.hormonal.explain --subject 5 --provider template   # offline, no API key
```

Inputs (defaults point at the B1 run): `data/processed/shap_importance_B1.csv`
(global drivers), `data/processed/oof_B1_conformal.parquet` (predictions +
conformal set / no-call), `data/processed/mcphases_daily.parquet` (hormone /
symptom / wearable trends). It builds a per-subject evidence pack, then generates
the report with **OpenAI gpt-4o** (the roadmap's OpenAI-credits path; set
`OPENAI_API_KEY`), falling back to Anthropic, then a deterministic template so a
real report is produced with no key at all. Every number comes from the evidence
pack — the LLM interprets, it does not invent. Output: `experiments/hormonal/reports/`.

## Backends: EBM, conformal, TabPFN

- **EBM** (`--backend ebm`, config `phase-clf-ebm.yaml`): glassbox. `ebm_importances.json`
  (term importances) is emitted for free — no SHAP. Slightly slower to fit than a
  boosted tree, but explanation cost is zero, so end-to-end (train + explain) it's competitive.
- **Conformal no-call** (`conformal:` block, any backend): LAC or APS prediction sets on the
  OOF probabilities. Reports empirical coverage (should meet `1 - alpha`), decisive
  (singleton) rate, and accuracy-when-decisive. LAC is the robust default; APS gives
  better class-conditional coverage on softer probabilities but over-inflates sets when
  the model is near-deterministic.
- **TabPFN** (`--backend tabpfn`, config `phase-clf-tabpfn.yaml`): no gradient training —
  `fit` stores context, `predict` is a forward pass. **TabPFN v2 gates its weights behind a
  one-time license**: set `TABPFN_TOKEN` (from https://ux.priorlabs.ai) before first run, or
  it cannot download the model. On CPU expect seconds/fold for this data size; a GPU is ~10-50x faster.

## Data schema (real mcPHASES names are the default)

The canonical column names ARE the real `mcphases_daily.parquet` names, defined
once in `data.py` (`NUMERIC_COLS`, `ORDINAL_COLS`, `HORMONE_COLS`, `PHASES`). The
`data:` config block just points at the file; no renaming needed. The synthetic
fallback emits the same names, so a no-data run is schema-consistent.

| group | columns (real mcPHASES) |
|---|---|
| keys | `id`, `study_interval`, `day_in_study` |
| label | `phase` ∈ {`Menstrual`, `Follicular`, `Fertility`, `Luteal`} (hormone-verified) |
| wearable (numeric) | `rhr_value, hr_bpm_mean, hrv_rmssd_mean, resp_full_sleep_breathing_rate, wtemp_temperature_diff_from_baseline, sleep_overall_score, stress_stress_score, steps_steps, cgm_glucose_value_mean/std`, … |
| symptoms (ordinal strings) | `flow_volume, cramps, bloating, moodswing, sorebreasts, fatigue, …` — encoded via `SEVERITY_MAP` (Not at all → Very High = 0…6) |
| hormones (regression targets) | `lh, estrogen, pdg` — Mira metabolites; `pdg` is sparse (~33%), so regression masks per-hormone NaNs |

LOSO groups by `id` (both study intervals of a subject stay together — no leakage).
Feature engineering runs within each `(id, study_interval)` segment: 3/5-day rolling
mean+std, day-1 delta, deviation from the segment's follicular baseline, plus
`days_since_bleed` and `is_weekend`. Missing values are filled within segment
(then global) so tree and glassbox backends all get finite inputs.

The raw 23 mcPHASES tables in `data/raw/mcphases/` are merged into this daily
table by the data-prep step (already done → `data/processed/mcphases_daily.parquet`).

## What it reports

Classification, out-of-fold over all LOSO folds: macro-F1 (vs SOTA 0.662),
accuracy, balanced accuracy, OvR AUROC, multiclass Brier. Plus:
- **Regular vs irregular** cycles separately — the regime prior work degrades on.
- **Conformal no-call**: coverage guarantee, decisive rate, accuracy-when-decisive.
- **Explanation**: EBM term importances (free) or SHAP (tree backends).

Regression: per-hormone MAE / RMSE / R² / Spearman, overall and by regularity.

## Artifacts (`experiments/hormonal/<name>/`)

`metrics.json`, `loso_folds.json`, `oof_predictions.parquet`, `MODEL_CARD.md`,
the saved final model, and `shap_importance.json` / `ebm_importances.json`.
Comparison run: `compare/comparison_report.md` + `comparison.json`.

## Files

- `configs/` — `phase-clf-baseline.yaml` (CatBoost), `phase-clf-ebm.yaml` (EBM+conformal),
  `phase-clf-tabpfn.yaml` (TabPFN), `compare.yaml` (head-to-head).
- `data.py` — load-or-synthesize, hormone targets, LOSO-safe feature engineering.
- `metrics.py` — classification metrics, threshold no-call, calibration-sensitive scoring.
- `conformal.py` — split-conformal prediction sets (LAC / APS) for the principled no-call.
- `regression.py` — hormone-level LOSO regression branch + metrics.
- `train.py` — classification LOSO CV, calibration, final model + explanation, tracking, model card.
- `compare.py` — runs all backends + the regression branch under one LOSO split, tabulates.

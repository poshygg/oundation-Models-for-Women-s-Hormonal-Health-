# Four-Phase Menstrual Cycle Classification on mcPHASES

**Hack Nation MIT — Challenge 05: Foundation Models for Women's Hormonal Health (Layer 02).**

A reproducible model and an open benchmark for inferring the four menstrual cycle phases —
menstrual, follicular, fertility (ovulatory), and luteal — from daily self-reported symptoms
and wearable physiology. Phase labels in mcPHASES come from at-home hormone assays, so the
hormone readings are treated as label sources and kept out of the feature set.

Evaluated leave-one-subject-out over 42 participants, reported with subject-bootstrap
confidence intervals. The model abstains (no-call) via conformal prediction when it is not
confident, rather than forcing a phase.

## What's here

```
benchmark/                 open benchmark: frozen LOSO split + model-agnostic CI scorer
  ├── eval/ci_harness.py     score any out-of-fold predictions with bootstrap CIs
  ├── eval/reference_model.py the two-stage reference model, end to end
  ├── splits/loso_folds.json  frozen 42-fold subject split
  ├── results/                reference out-of-fold predictions
  └── MODEL_CARD.md
ml/mcphases/               data assembly + pipeline (features, HSMM, fertility detector, conformal, SHAP)
ai/hormonal/               model code (classification + hormone-regression branches)
docs/challenges/05-womens-hormonal-health/technical-report.md   full write-up
notebooks/eda_figures/     exploratory analysis figures
data/                      folder structure only; raw mcPHASES is not redistributed (see below)
```

## Results (out-of-fold, LOSO, N=42)

| model | macro-F1 [95% CI] | accuracy | fertility F1 |
|---|---|---|---|
| base classifier + HSMM | 0.654 [0.601, 0.701] | 0.671 | 0.498 |
| two-stage (+ fertility detector) | **0.667 [0.620, 0.709]** | 0.678 | **0.531** |

Backend choice is not the lever — CatBoost, XGBoost, and LightGBM land within ~0.01 of each
other. The gains come from data alignment, the cycle-position feature, and the dedicated
fertility detector. Full details in the [technical report](docs/challenges/05-womens-hormonal-health/technical-report.md).

## Reproducing

```bash
python ml/mcphases/build_master.py            # assemble the aligned daily table
python benchmark/eval/reference_model.py       # train + write out-of-fold predictions
python benchmark/eval/ci_harness.py            # score with bootstrap CIs
```

Seeds are fixed (42) and the split is frozen, so the numbers above regenerate.

## Data

This repository redistributes **no raw mcPHASES data**. mcPHASES is distributed by PhysioNet
under its own credentialed-access license; obtain the raw data directly from
https://physionet.org/content/mcphases/ and place it under `data/raw/mcphases/`.

## License

MIT (see `LICENSE`) — covers this repository's own code, evaluation harness, splits, and
documentation, not the underlying dataset.

## Note

Research and infrastructure, not a medical device. No diagnostic claim is made. The cohort is
small (42 young adults, one region), so external validity to broader populations is untested.

# mcPHASES 4-Phase Cycle Benchmark (leakage-audited)

A reproducible, **leakage-audited** benchmark and evaluation harness for 4-phase
menstrual-cycle classification from multimodal wearables + self-report on the
[mcPHASES](https://physionet.org/content/mcphases/) dataset.

The field has a reproducibility / ground-truth gap: wearable cycle-tracking papers
rarely publish standardized splits, and it is easy to leak the label into a
cycle-position feature. This benchmark exists so researchers can **compare methods on
identical honest splits, reproduce results, and extend the work** instead of rebuilding
the same foundation in isolation.

## What's here

| File | Purpose |
|---|---|
| `splits/loso_folds.json` | Frozen leave-one-subject-out split (42 folds, one held-out subject each) |
| `eval/ci_harness.py` | Model-agnostic **subject-bootstrap 95% CI** on macro-F1 / accuracy / per-class F1 |
| `eval/reference_model.py` | Honest reference model (CatBoost + HSMM), regenerates the split + OOF |
| `results/oof_reference.parquet` | Reference out-of-fold predictions (input to the CI harness) |
| `MODEL_CARD.md` | Reference-model card: data, training, metrics, limitations, leakage audit |
| `LICENSE` | MIT (covers this benchmark's code/splits/docs; **not** the mcPHASES raw data) |

## Task

Per subject-day, predict the menstrual phase ∈ {Menstrual, Follicular, Fertility, Luteal}
(hormone-verified labels). **Protocol: leave-one-subject-out** — train on 41 subjects,
test on the held-out one. Primary metric: **macro-F1** (4 balanced classes), reported
with a subject-bootstrap 95% CI. Reference point: published SOTA = 0.662 / 67.6% acc.

## Data alignment (comparable to the SOTA paper)

We use the mcPHASES **2022 interval** (the 2024 interval is ~99% empty on self-report),
drop a 14-day feature warmup, and exclude cycle stretches with ≥5 consecutive
symptom-missing days. This yields a **paper-replication view** (2,965 rows / 42 subjects,
≈ the paper's 2,983 / 41) and a **multimodal view** (3,110 rows — 145 extra days kept
because wearables are still present when symptoms are missing). The reference model uses
the multimodal view. Raw mcPHASES data is **not redistributed** here (PhysioNet license);
only subject-level split indices and derived artifacts.

## Reference model & results (leakage-free, no selection on test)

**Two-stage:** a 4-class CatBoost over multimodal + physiology features, plus a dedicated
binary **Fertility (ovulation-window) detector**; their out-of-fold Fertility probabilities
are blended at a fixed 0.5 weight, then decoded with a per-fold explicit-duration HSMM.

| Metric | Score | 95% CI (subject bootstrap) |
|---|---|---|
| **macro-F1** | **0.667** | **[0.620, 0.709]** |
| accuracy | 0.678 | [0.631, 0.721] |
| F1 Menstrual | 0.747 | [0.692, 0.797] |
| F1 Follicular | 0.627 | [0.576, 0.674] |
| F1 Fertility | 0.531 | [0.464, 0.595] |
| F1 Luteal | 0.761 | [0.714, 0.803] |

Ablation (single-stage, no Fertility detector): macro-F1 **0.654**, Fertility F1 0.498.
The two-stage detector lifts the ovulation window **+0.033** (0.498 → 0.531) — the class
every prior method fails on — and macro-F1 **+0.013**.

**Verdict:** the 95% CI contains the published SOTA (0.662), with the point estimate
slightly above it → **statistically on par with SOTA**, achieved honestly (leakage-free
features, per-fold HSMM durations, a fixed blend weight — no hyperparameter tuned on the
reported split). At N=42 a single point (0.667 vs 0.662) is within noise, so we report
CIs, not bare points.

## ⚠️ Leakage audit (a methods contribution)

A naive "days since last period" feature computed from the **phase label** inflated
macro-F1 from an honest **0.52 to a leaked 0.67** (that one feature was 26% of SHAP
importance). This benchmark's cycle-position feature is derived **only from self-reported
menstrual flow** (`flow_volume`), never the label, and is verified causal (past-only).
We recommend every submission run the same check: *if a single feature dominates
importance and traces back to the label, it is leakage.*

## Reproduce

```bash
# 1. build the aligned master table (from a local mcPHASES daily table)
python ml/mcphases/build_master.py
# 2. train the reference model -> regenerates splits + OOF
python benchmark/eval/reference_model.py
# 3. evaluate with confidence intervals
python benchmark/eval/ci_harness.py --oof benchmark/results/oof_reference.parquet
```

`ci_harness.py` is model-agnostic: point any method's OOF file (columns `id, phase, pred`)
at it to get a CI comparable to the reference.

## What this leaves behind (Foundation Value)

- **Frozen honest splits** + a **model-agnostic CI harness** → apples-to-apples method comparison.
- A documented **leakage audit** the community can reuse to avoid a common pitfall.
- A reproducible **reference model** (SOTA-par, leakage-free) others can extend or beat.

## Limitations

N=42 subjects, young (18–29) and demographically narrow cohort, 2022 interval only.
Fertility (ovulation window) is the hardest class (~0.50 F1) for every method. The
cycle-position feature depends on self-reported flow, so performance degrades for users
who do not log menstruation — the wearable channels partially mitigate this but do not
eliminate it. Not a diagnostic device; no clinical claims.

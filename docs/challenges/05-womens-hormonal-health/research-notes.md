# Challenge 05 — Research Notes (Layers 1 & 2)

Prior work survey for the "search papers → time budget → code" build order. Sourced 2026-07 via web search; see the URL list at the bottom.

---

## Part A — Challenge structure (English)

**Goal:** Build one *reusable building block* toward open AI infrastructure for women's hormonal health. Not a full foundation model in a weekend — one dataset, benchmark, model, or app that future researchers can immediately build on. **Open-source licensing is a scored criterion, not optional.**

**Three layers (pick one as the main contribution):**

| Layer | What it is | Deliverable |
|---|---|---|
| **01 Data & Benchmark** | Standardized multimodal dataset (wearables + lab + imaging + symptoms + longitudinal signals) with documented train/val/test splits and a transparent evaluation method | An open dataset/benchmark future researchers can use immediately |
| **02 AI Model** | A focused model optimized for reproducibility, scientific rigor, and explainability over size (hormone-level / hormonal-state prediction, e.g. early/late menopause onset) | A reproducible model others can extend and combine |
| **03 Application** | Solve one clearly defined problem on top of the above (symptom tracker, digital hormone journal, personalized insight) | Measurable, meaningful impact for women |

**Evaluation:** Women's Health Impact · Technical Excellence (rigor / reproducibility / scalability) · Foundation Value (does it leave behind reusable infrastructure).

**Provided data:** mcPHASES (PhysioNet — Fitbit, CGM, hormones, cycle, sleep, symptoms), NHANES (CDC — reproductive health, thyroid hormones, labs).

**Strong vs weak (from the brief):** Strong = open, reusable, one problem solved rigorously, reproducible code. Weak = isolated app with no reusable contribution, undocumented proprietary data, or unsupported medical claims.

---

## Part B — Layer 01 precedents: datasets & benchmarks (English)

The field's core problem is a **reproducibility / ground-truth gap**: most wearable menstrual-health studies use self-reported cycle labels (noisy) and do not publish code or data splits.

- **mcPHASES (PhysioNet, 2025)** — the closest precedent and our provided dataset. N=42 Canadian menstruators, two 3-month intervals, Fitbit Sense + Dexcom G6 CGM + Mira hormone kit (LH/E3G/PdG) + daily symptom diary. 23 tables, hormone-verified 4-phase labels, **no imputation** (real-world). It exists *because* prior datasets lacked longitudinal, ambulatory hormone ground truth.
- **NHANES (CDC)** — large cross-sectional public survey (reproductive health, thyroid hormones, labs, nutrition, demographics). Good for population-level disease-risk benchmarks; not longitudinal.
- **Field pattern** — published wearable-menstrual models rarely release standardized splits, so results are hard to compare. A documented, subject-level split + evaluation harness is itself a valuable Layer-01 contribution (directly hits "Foundation Value").

**Takeaway for us:** the highest-leverage Layer-01 move is *not* collecting new data — it's turning mcPHASES into a **documented benchmark** (subject-level splits, defined task, transparent metrics) that others can reproduce.

---

## Part C — Layer 02 precedents: models & architectures (English)

### C1. Menstrual phase / ovulation classification

| Study | Input → output | Model | Result |
|---|---|---|---|
| Empatica observational (2019) | skin temp, EDA, IBI, HR → 3 phases | **Random Forest** | 87% acc, AUROC 0.96; fertile window 90% |
| Wrist temp + HR (2022) | temp + HR → fertile window / menstruation | ML algorithm | regular 85.5%, irregular 79.9%, first-day 61%, menstruation 51% |
| Sleeping HR (2025) | night HR → phase + ovulation day | ML classifier | free-living 3-phase |
| Multimodal DL (2024) | temp, HRV, EDA, sleep → phases | **TCN + LSTM** | real-time multimodal |
| Ovulation window | ECG HRV + temp | **LightGBM** | ovulation window |
| Circular-statistics (2023) | wearable signals | circular stats + ARIMA | periodicity analysis |

### C2. Hormone-level regression (predicting the value, not just the phase)

| Study | Input → output | Model | Result |
|---|---|---|---|
| FET biosensor digital twin (2026) | sensor time/morphology features → **estradiol level** | **CatBoost** (best) | real-time estradiol |
| Continuous glucose prediction (2025) | multimodal wearables → glucose | linear/ridge/RF/**XGBoost** | tree > linear, XGBoost R²=0.73 |

### C2b. ⭐ DIRECT SOTA on mcPHASES — the benchmark we must beat (found 2026-07)

- **"Self-Reported Symptoms Enable Four-Phase Menstrual Cycle Classification with Hormonally Validated Labels"** (medRxiv 2026, `10.64898/2026.03.31.26349766`) — built *on mcPHASES itself*. **CatBoost + Hidden Semi-Markov Model (HSMM)** hybrid: CatBoost captures non-linear symptom patterns from **83 engineered features**; HSMM imposes biology (cyclic phase order + Gaussian phase-duration priors) for temporally coherent output.
  - **Split:** leave-one-subject-out (LOSO) on 41 participants → no subject leakage (this is the honest, hard split).
  - **Result:** **67.6% accuracy, macro-F1 0.662** (4-phase).
  - **Key strategic point:** they deliberately use **self-report symptoms ONLY, no wearables** ("...tracking *without* wearables: implications for equitable digital health"). → **This is our opening: adding Fitbit temp/HR/HRV/sleep + CGM on top of their 83 symptom features to beat 67.6% under the same LOSO split is a clean, defensible contribution.**
- **Dataset descriptor** (Nature Scientific Data 2026, `s41597-026-06805-3`) — technical validation confirms our feature engineering: nightly wrist temp & resting HR show phase-dependent variation (rmANOVA p<.001); LH mid-cycle peak, estrogen pre-ovulation rise, PdG luteal peak all match endocrinology. Caveats: N=42 (only 20 returned for Interval 2), 59.5% East/SE Asian mean age 20.9, **no imputation**, Interval 2 dropped glucose + some surveys, proprietary device algorithms.

### C3. Reproducible + explainable + calibrated (the Layer-02 template)

- **PCOS explainable & fair AI (arXiv 2511.11636, 2025)** — near-perfect blueprint for what the challenge wants: **gradient boosting** model + **SHAP** explainability + **probability calibration** (reliability curves) + **subgroup equity audit** + interactive deployment. This is the exact 4-part shape (SHAP + calibration + fairness + reproducibility) to copy.

### C4. Physiological basis (grounds the feature engineering)

- Post-ovulation progesterone raises basal body temperature **+0.28–0.56 °C**; resting HR, respiratory rate, and HRV are elevated in the luteal phase. Feature engineering has literature backing — no need to guess.

### C5. Convergent finding

Across regression and classification, **gradient-boosted trees (RF / XGBoost / CatBoost / LightGBM) repeatedly win** on this tabular, small-N, multimodal data — deep nets (TCN/LSTM) come second and need more data. On mcPHASES (N=42), a boosted-tree baseline is both the literature-backed and the pragmatic choice.

---

## Part D — Recommended model structure (English)

```
Input: mcPHASES multimodal features
  (Fitbit HR / temp / HRV / sleep / stress + CGM glucose + self-report symptoms)
        |
        v  ## per-subject windowing + SUBJECT-LEVEL split (prevent leakage)
Feature engineering (physiology-grounded: luteal HR/temp rise, etc.)
        |
        v
Model: Gradient Boosting (XGBoost / CatBoost / LightGBM)
        |
        |-- classification: 4 phases (menses / late-follicular / ovulation / luteal)
        '-- regression: LH / E3G / PdG levels (mcPHASES has hormone-verified labels)
        |
        v
Explainability: SHAP  +  Uncertainty: CONFORMAL PREDICTION → no-call (adopted upgrade)
        |
        v
Evaluation: balanced accuracy / AUROC / Brier, reported separately for regular vs irregular cycles
        |
        v
Open-source release: HF model card + reproducible splits + code
```

---

## Part D2 — Novel architecture review (is CatBoost+HSMM too derivative?) — added 2026-07

CatBoost+HSMM **is** the published mcPHASES SOTA, so as a headline it reads as replication. Reviewed 2024–2026 frontier architectures for a fresher-yet-feasible core. Ranked by (novelty × 24h-feasibility × fit for N=42, ~5.6k rows, ~50% missing).

**Tier A — novel AND swap-in feasible (recommended headline):**
- ⭐ **TabPFN v2 / v2.5** (Nature 2024, `s41586-024-08328-6`) — a *tabular foundation model*: a transformer pre-trained on ~130M synthetic datasets that does in-context learning, **no gradient training at inference**. Literally a "foundation model" (matches the challenge title). Beats tuned XGBoost/CatBoost on datasets **<10k rows** (our ~5.6k fits), **100% win rate vs default XGBoost <10k**, needs little/no feature engineering. **One-line swap for CatBoost.** Also has a v2 time-series forecasting mode (arXiv `2501.02945`). Best novelty-per-risk on the board.

**Tier B — novel, higher effort, strong "Foundation Value" pitch:**
- **Observation-triplet transformers (STraTS / SCANE)** — represent each measurement as a `(feature, time, value)` token, **imputation-free**. Directly fits mcPHASES' irregular + no-imputation design instead of forcing a dense daily matrix. More architecturally novel; more implementation risk.
- **mTAND / multi-time attention** — continuous-time attention over irregular samples; classic irregular-clinical-TS approach.
- **Wearable foundation-model embeddings** — extract pretrained representations of Fitbit HR/PPG/accel signals from an open physiological FM, then a light head (or feed into TabPFN). Open candidates: **PaPaGei**, **AnyPPG** (arXiv `2511.01747`), **HiMAE** (`2510.25785`); Google **LSM/SensorFM** (ICLR 2025 `2410.13638`) is strongest but weights aren't openly usable. This is the most on-theme "reusable infrastructure" angle.

**Tier C — avoid as core (novelty ≠ fit):**
- **Full TCN+LSTM+attention multimodal DL** (RealTime menstrual DL 2024) needs ~6,000 cycles; we have 42 subjects → overfit + demo risk. Cite as related work, don't build.
- **Training our own foundation model** — infeasible in 24h.

**Recommended synthesis:** keep **CatBoost = reproducible baseline** (confirm 67.6% under LOSO), but make the **headline = TabPFN v2.5** (foundation-model framing, small-N strength) and, if time allows, **wearable-FM embeddings as features** (Foundation Value). Honest caveat for the pitch: at N=42 the *split discipline and multimodal fusion* drive the score more than the architecture — deep sequence models will likely NOT beat TabPFN/trees here. Novelty should be the framing, not a bet against the data size.

---

**Our 3 differentiators (attacking the field's common weaknesses):**
1. **Hormone-verified labels** — most prior work uses noisy self-reported labels; mcPHASES has Mira hormone ground truth.
2. **Honest irregular-cycle reporting + no-call** — prior models degrade to 50–80% on irregular cycles and gloss over it; we report separately and abstain.
3. **Fully reproducible & open** — the field rarely releases code/splits (reproducibility crisis) → directly earns Foundation Value.

---

## Part E — 한국어 번역

### 챌린지 구조 요약
**목표:** 여성 호르몬 건강을 위한 열린 AI 인프라에 기여하는 **재사용 가능한 블록 하나**를 만든다. 주말에 거대 파운데이션 모델을 완성하는 게 아니라, 미래 연구자가 즉시 갖다 쓸 데이터셋·벤치마크·모델·앱 하나. **오픈소스 공개는 선택이 아니라 채점 항목.**

**세 레이어(하나를 메인으로 선택):**
- **01 데이터 & 벤치마크** — 웨어러블·검사실·영상·증상·종단신호를 결합한 표준화 멀티모달 데이터셋 + 문서화된 분할·투명한 평가. → 즉시 쓸 수 있는 공개 데이터셋/벤치마크.
- **02 AI 모델** — 크기보다 재현성·엄밀성·설명가능성에 최적화한 집중 모델(호르몬 수치/상태 예측, 예: 폐경 조기/후기 신호). → 재현·확장 가능한 모델.
- **03 애플리케이션** — 위 위에서 문제 하나 해결(증상 트래커·호르몬 일지·개인화 인사이트). → 측정 가능한 임팩트.

**평가:** 여성 건강 임팩트 · 기술적 완성도(엄밀성·재현성·확장성) · 파운데이션 가치(재사용 인프라를 남기는가).
**제공 데이터:** mcPHASES(PhysioNet), NHANES(CDC).

### Layer 01 선례 — 데이터셋·벤치마크
이 분야의 근본 문제는 **재현성·정답라벨 공백**: 대부분의 웨어러블 월경 연구가 잡음 많은 자가보고 라벨을 쓰고 코드·분할을 공개하지 않음.
- **mcPHASES (PhysioNet, 2025)** — 가장 가까운 선례이자 제공 데이터. N=42, 두 번의 3개월, Fitbit + CGM + Mira 호르몬(LH/E3G/PdG) + 증상 일지. 호르몬 검증 4단계 라벨, 보간 없음. 기존 데이터에 종단·앰뷸러토리 호르몬 정답이 없어서 탄생.
- **NHANES (CDC)** — 대규모 횡단면 공개 조사. 인구집단 질환위험 벤치마크에 적합, 종단 아님.
- **시사점:** Layer 01의 최고 레버리지는 새 데이터 수집이 아니라, mcPHASES를 **피험자 단위 분할 + 태스크 정의 + 투명한 지표**를 갖춘 재현 가능한 벤치마크로 만드는 것 (= Foundation Value 직격).

### Layer 02 선례 — 모델·구조
- **단계/배란 분류:** Empatica RF(3단계 87%, AUROC 0.96, 가임기 90%), 손목온도+심박(규칙 85.5%/불규칙 79.9%/첫날 61%/월경 51%), 수면심박 분류, 멀티모달 딥러닝 TCN+LSTM, 배란 LightGBM, 원형통계+ARIMA.
- **호르몬 수치 회귀:** FET 바이오센서 디지털트윈 → 에스트라디올(**CatBoost** 최고), 연속혈당 예측(**XGBoost** R²=0.73, 트리>선형).
- **재현·설명·보정 템플릿:** PCOS 설명가능·공정 AI(arXiv 2511.11636) — **gradient boosting + SHAP + 확신도 보정 + 하위집단 공정성 감사 + 인터랙티브 배포.** 우리가 그대로 복제할 4요소 구조.
- **생리학적 근거:** 배란 후 프로게스테론이 기초체온 **+0.28~0.56°C** 상승, 황체기에 안정시 심박·호흡률·HRV 상승 → 피처 엔지니어링에 문헌 근거 있음.
- **수렴 결론:** 회귀·분류 모두에서 **gradient boosting 트리(RF/XGBoost/CatBoost/LightGBM)가 반복 승리** — 딥러닝은 그다음, 데이터 더 필요. N=42인 mcPHASES엔 트리 baseline이 문헌적·실용적 정답.

### 권장 모델 구조 (한국어)
입력(mcPHASES 멀티모달 피처) → 피험자 단위 윈도잉·**피험자 단위 분할(누수 방지)** → 생리학 근거 피처 엔지니어링 → **Gradient Boosting** (분류: 4단계 / 회귀: LH·E3G·PdG) → **SHAP 설명 + [채택 업그레이드] Conformal Prediction 기반 no-call(분포무관 커버리지 보장)** → 규칙적/불규칙 주기 분리 평가(balanced acc / AUROC / Brier) → **오픈소스 공개(HF 모델카드 + 재현 가능한 분할·코드)**.

> **파이프라인 확정(2026-07):** 검토한 교체 후보(TabPFN/EBM 분류기, CRF, 자동 TS 피처, Conformal) 중 **06 불확실성 단계만 Conformal Prediction으로 교체**하기로 결정. 나머지(CatBoost·HSMM·SHAP·LOSO)는 SOTA 재현·비교 가능성을 위해 유지. Conformal은 model-agnostic wrapper(~50줄)라 CatBoost 위에 그대로 얹히고, LOSO 유지를 위해 **conformal 보정 분위수는 held-out 피험자에서** 계산해야 함.

**차별화 3가지:** ① 호르몬 검증 라벨 기반 평가(선행연구 대부분 자가보고), ② 불규칙 주기 정직 보고 + no-call, ③ 완전 재현·공개(이 분야 고질적 재현불가 문제 공략 = Foundation Value).

---

## Sources
- ⭐ DIRECT SOTA on mcPHASES — CatBoost+HSMM, LOSO 67.6% acc / 0.662 macro-F1 (medRxiv 2026): https://www.medrxiv.org/content/10.64898/2026.03.31.26349766v2.full
- ⭐ Same work, self-report-only framing (Research Square 2026): https://www.researchsquare.com/article/rs-9497159/v1
- mcPHASES dataset descriptor (Nature Scientific Data 2026): https://www.nature.com/articles/s41597-026-06805-3  ·  PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC13003092/
- mcPHASES (PhysioNet, 2025): https://physionet.org/content/mcphases/1.0.0/
- ML-based menstrual phase identification (npj Women's Health, 2025): https://www.nature.com/articles/s44294-025-00078-8
- Sleeping-HR phase classification (ScienceDirect, 2025): https://www.sciencedirect.com/science/article/pii/S0010482525000551
- Wearable sensors → fertile window (PubMed, 2019): https://pubmed.ncbi.nlm.nih.gov/30998226/
- BBT + HR cycle tracking + ML (PMC, 2022): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9375297/
- Multimodal DL for menstrual/ovulatory prediction (2024): https://www.researchgate.net/publication/392268220
- FET biosensor digital twin, estradiol / CatBoost (Wiley, 2026): https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202500950
- Continuous glucose prediction, XGBoost (PMC, 2025): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12115526/
- Explainable & fair PCOS AI, SHAP + calibration + equity (arXiv, 2025): https://arxiv.org/pdf/2511.11636
- Menstrual cycle PK/PD of drugs (Clinical Pharmacokinetics): https://link.springer.com/article/10.2165/00003088-199834030-00003

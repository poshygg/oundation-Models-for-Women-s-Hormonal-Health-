# Baseline Plan — Challenge 05 / Layer 02 (Menstrual-Phase Model)

Two baselines, in order. **B0** reproduces the one published result on this dataset so our
numbers are trustworthy and comparable. **B1** is the version we actually want to ship: the same
pipeline with two deliberate changes. Everything not listed as "changed" is kept identical on
purpose — that identity is what makes the comparison fair.

---

## English

### B0 — Base baseline (reproduce the published SOTA)

**Purpose.** Prove our harness is correct and get a fair yardstick before we claim any improvement.
The only published model on mcPHASES (CatBoost + HSMM, self-report symptoms only) reports
**67.6% accuracy / 0.662 macro-F1** under leave-one-subject-out. If we can't reproduce that, no later
number means anything.

| Stage | B0 choice |
|---|---|
| Data | Self-report symptom features **only** (match the paper) |
| Split | Leave-One-Subject-Out (LOSO) — hold out whole participants |
| Features | ~83 engineered symptom features (Likert values + rolling stats) |
| Model | **CatBoost**, 4-phase classifier (Menstrual / Follicular / Fertility / Luteal) |
| Temporal | **HSMM** — cyclic phase order + duration priors |
| Uncertainty | Simple confidence-threshold no-call |
| Explainability | **SHAP** (TreeSHAP) |
| Target | Match ≈ **67.6% acc / 0.662 macro-F1** |

### B1 — Target baseline (what we change to)

**Purpose.** Beat B0 honestly and add scientific rigor, without breaking comparability.
**Exactly two changes** from B0:

1. **Add multimodal features.** Layer Fitbit temperature / resting HR / HRV / sleep / stress and
   Dexcom CGM on top of the symptom features. This is the intended source of lift over 67.6% —
   the published model deliberately excluded wearables, so this is our clean opening.
2. **Swap the uncertainty stage → Conformal Prediction.** Replace the arbitrary confidence
   threshold with **split conformal prediction**: prediction sets with a guaranteed coverage level
   (the true phase is in the set ≥ 1−α of the time). Singleton set → confident call; empty or
   multi-phase set → **no-call**. Use the class-conditional (Mondrian) variant so the guarantee
   holds separately for regular vs irregular cycles.

Everything else — CatBoost, HSMM, SHAP, LOSO, metrics — is **unchanged**.

| Stage | B0 | B1 (changed?) |
|---|---|---|
| Data | symptoms only | **+ Fitbit + CGM (changed)** |
| Split | LOSO | LOSO (same) |
| Features | ~83 symptom | **+ physiology-grounded wearable features (changed)** |
| Model | CatBoost | CatBoost (same) |
| Temporal | HSMM | HSMM (same) |
| Uncertainty | threshold no-call | **Conformal Prediction → no-call (changed)** |
| Explainability | SHAP | SHAP (same) |
| Success | match 67.6% | **macro-F1 > 0.662 under identical LOSO + valid conformal coverage** |

**Why only these two.** More novel swaps were reviewed (TabPFN, EBM, CRF, Catch22/MiniRocket).
We parked them: at N=42 the split discipline and multimodal fusion drive the score more than the
architecture, and changing the model would break the direct 67.6% comparison. Conformal prediction
was the cheapest, highest-rigor win — a ~50-line model-agnostic wrapper that sits on CatBoost unchanged.

**What must not change (and why).** LOSO split, the four metrics (balanced accuracy / macro-F1 /
AUROC / Brier), CatBoost, HSMM, and SHAP stay fixed so B1 is directly comparable to both B0 and the
published SOTA. If we move these, the headline "we beat 67.6%" stops being meaningful.

**Implementation guardrail.** Fit the conformal calibration quantile on **held-out subjects**, not
held-out days — otherwise the coverage guarantee breaks under LOSO.

**Parked for later (not now).** TabPFN v2.5 (tabular foundation model, small-N strength), EBM
(glass-box, explanation-is-the-model), CRF (discriminative temporal), Catch22/MiniRocket (automated
features). Revisit only if B1 is finished with time to spare.

---

## 한국어

### B0 — 원래(기본) 베이스라인: 공개 SOTA 재현

**목적.** 개선을 주장하기 전에, 우리 실험 파이프라인이 맞다는 걸 증명하고 공정한 기준선을 확보한다.
mcPHASES로 나온 유일한 논문(CatBoost + HSMM, 자가보고 증상만)은 leave-one-subject-out에서
**정확도 67.6% / macro-F1 0.662**를 보고한다. 이걸 재현 못 하면 이후 어떤 숫자도 의미가 없다.

| 단계 | B0 선택 |
|---|---|
| 데이터 | 자가보고 증상 피처 **only** (논문과 동일) |
| 분할 | Leave-One-Subject-Out(LOSO) — 피험자 통째로 홀드아웃 |
| 피처 | 증상 기반 ~83개 (Likert 값 + 롤링 통계) |
| 모델 | **CatBoost** 4단계 분류 (월경/난포/가임/황체) |
| 시간 후처리 | **HSMM** — 주기 순서 + 지속시간 사전확률 |
| 불확실성 | 단순 신뢰도 임계값 no-call |
| 설명 | **SHAP**(TreeSHAP) |
| 목표 | ≈ **67.6% / 0.662 macro-F1** 재현 |

### B1 — 바꾸고 싶은(목표) 베이스라인

**목적.** 비교 가능성을 깨지 않으면서 B0을 정직하게 이기고 과학적 엄밀성을 더한다.
B0에서 **딱 두 가지만** 바꾼다:

1. **멀티모달 피처 추가.** 증상 피처 위에 Fitbit 체온/안정시 심박/HRV/수면/스트레스 + Dexcom CGM을
   얹는다. 이게 67.6%를 넘기는 **의도된 성능 상승원**이다 — 논문이 웨어러블을 일부러 뺐기 때문에
   우리에겐 깨끗한 빈틈이다.
2. **불확실성 단계 → Conformal Prediction 교체.** 임의의 신뢰도 임계값 대신 **split conformal
   prediction**을 쓴다: 커버리지가 보장된 예측집합(참 phase가 집합에 ≥ 1−α 확률로 포함)을 만들고,
   단일 phase면 확신 예측, 비었거나 여러 phase면 **no-call**. class-conditional(Mondrian)로 규칙/
   불규칙 주기에 각각 보장이 성립하게 한다.

나머지 — CatBoost, HSMM, SHAP, LOSO, 지표 — 는 **그대로**.

| 단계 | B0 | B1 (변경?) |
|---|---|---|
| 데이터 | 증상만 | **+ Fitbit + CGM (변경)** |
| 분할 | LOSO | LOSO (동일) |
| 피처 | 증상 ~83 | **+ 생리학 근거 웨어러블 피처 (변경)** |
| 모델 | CatBoost | CatBoost (동일) |
| 시간 후처리 | HSMM | HSMM (동일) |
| 불확실성 | 임계값 no-call | **Conformal Prediction → no-call (변경)** |
| 설명 | SHAP | SHAP (동일) |
| 성공 기준 | 67.6% 재현 | **동일 LOSO에서 macro-F1 > 0.662 + conformal 커버리지 유효** |

**왜 이 둘만.** 더 새로운 교체안(TabPFN, EBM, CRF, Catch22/MiniRocket)도 검토했지만 보류했다.
N=42에선 구조보다 **분할 규율과 멀티모달 융합**이 점수를 더 좌우하고, 모델을 바꾸면 67.6%와의
직접 비교가 깨진다. Conformal은 **가장 값싸고 엄밀성 점수가 큰 한 수** — CatBoost 위에 그대로
얹히는 ~50줄짜리 model-agnostic wrapper다.

**절대 바꾸지 말 것(이유).** LOSO 분할, 네 가지 지표(balanced accuracy / macro-F1 / AUROC / Brier),
CatBoost, HSMM, SHAP는 고정한다. 그래야 B1이 B0·공개 SOTA와 직접 비교된다. 이걸 건드리면
"우리가 67.6%를 이겼다"는 헤드라인 자체가 무의미해진다.

**구현 가드레일.** conformal 보정 분위수는 held-out "일(day)"이 아니라 held-out "피험자"에서
계산한다 — 아니면 LOSO에서 커버리지 보장이 깨진다.

**나중으로 보류(지금 아님).** TabPFN v2.5(tabular 파운데이션 모델, 소규모 N 강점), EBM(유리상자,
설명이 곧 모델), CRF(판별적 시간 모델), Catch22/MiniRocket(자동 피처). B1을 끝내고 시간이 남을
때만 다시 검토.

---

## Justification — why adopt Conformal Prediction (the one new component) / 왜 Conformal Prediction을 채택하나

### English

**Thesis.** The one new piece in B1 is worth it because it converts our "no-call" from a number we
guessed into a **statistical guarantee we can prove** — for almost no cost and no extra model.

**The problem it fixes.** B0's abstention uses an arbitrary confidence threshold (say 0.7). That number
is a heuristic: it promises nothing, and it is brittle exactly where it matters — irregular cycles,
where every published model silently drops to 50–60%. A clinician or judge cannot trust "confidence 0.7."

**What Conformal Prediction gives instead.** Split conformal wraps the CatBoost probabilities and, from a
held-out calibration set, returns a **prediction set** with a *distribution-free, finite-sample coverage
guarantee*: at a 90% target, the true phase is inside the set ≥ 90% of the time — provably, without
assuming any data distribution. Singleton set → confident call; empty or multi-phase set → **no-call**.
It is **model-agnostic** and sits on top of CatBoost unchanged.

**Why it wins on the scorecard.**
- **Technical Excellence / rigor** — we replace a heuristic with a *mathematical* guarantee; this is the single most rigorous thing a small-N model can claim.
- **Women's Health Impact** — safe abstention means we never hand a woman a confident-but-wrong phase call on an irregular cycle; we say "unsure" honestly.
- **Foundation Value** — it is a reusable, principled uncertainty layer any future model on this data can adopt.

**Cost / risk.** ~50 lines, no retraining, no second model, minutes to add. Effort and demo risk are near zero.

**Honest caveats to say out loud (don't hide them).**
- Conformal only *quantifies* uncertainty — it does **not** raise accuracy by itself. Accuracy comes from the multimodal features; conformal makes the abstention trustworthy.
- The guarantee assumes exchangeability, so under LOSO we **calibrate on held-out subjects, not days**.
- Use the class-conditional (Mondrian) variant so coverage is honest per phase and for regular vs irregular cycles.
- If the model is weak, sets get larger — that is the method being *honest*, not a bug; report the accuracy-vs-coverage curve.

**One-sentence pitch.** "Our model knows when it doesn't know: instead of a hand-picked threshold, we use
conformal prediction to abstain with a provable coverage guarantee — especially on the irregular cycles
where every prior model quietly fails."

### 한국어

**핵심 주장.** B1에서 새로 넣는 한 조각(Conformal Prediction)이 값어치를 하는 이유: 우리의 "no-call"을
*우리가 임의로 고른 숫자*에서 **증명 가능한 통계적 보장**으로 바꿔주기 때문 — 거의 공짜로, 별도 모델 없이.

**해결하는 문제.** B0의 기권은 임의의 신뢰도 임계값(예: 0.7)을 쓴다. 이 숫자는 휴리스틱이라 아무것도
보장하지 못하고, 정작 중요한 곳 — 모든 공개 모델이 조용히 50~60%로 떨어지는 불규칙 주기 — 에서 취약하다.
임상의나 심사위원은 "신뢰도 0.7"을 신뢰할 수 없다.

**Conformal Prediction이 대신 주는 것.** split conformal은 CatBoost 확률을 감싸서, held-out 보정셋으로부터
**분포무관·유한표본 커버리지 보장**을 갖는 **예측집합**을 낸다: 목표 90%면 참 phase가 집합 안에 ≥90%로
들어간다 — 어떤 분포 가정도 없이 증명적으로. 단일 phase면 확신 예측, 비었거나 다중이면 **no-call**.
**model-agnostic**이라 CatBoost를 그대로 두고 얹힌다.

**채점표에서 이기는 이유.**
- **기술적 완성도/엄밀성** — 휴리스틱을 *수학적* 보장으로 대체. 소규모 N 모델이 내세울 수 있는 가장 엄밀한 주장.
- **여성 건강 임팩트** — 안전한 기권 = 불규칙 주기에서 "확신에 찬 오답"을 여성에게 주지 않고 정직하게 "모르겠다"고 말한다.
- **파운데이션 가치** — 이 데이터 위 어떤 미래 모델도 갖다 쓸 수 있는 재사용 가능한 원칙적 불확실성 레이어.

**비용/리스크.** ~50줄, 재학습 없음, 두 번째 모델 없음, 추가에 몇 분. 노력·데모 리스크 거의 0.

**정직하게 밝혀야 할 한계(숨기지 말 것).**
- Conformal은 불확실성을 *정량화*할 뿐, 그 자체로 정확도를 **올리지 않는다.** 정확도는 멀티모달 피처에서 나오고, conformal은 기권을 신뢰 가능하게 만든다.
- 보장은 교환가능성을 가정하므로 LOSO에선 **held-out 일이 아니라 held-out 피험자에서 보정**한다.
- class-conditional(Mondrian)로 phase별·규칙/불규칙별 커버리지가 정직하게 성립하게 한다.
- 모델이 약하면 집합이 커진다 — 이는 버그가 아니라 방법이 *정직한* 것. accuracy-vs-coverage 곡선으로 보고.

**한 줄 피치.** "우리 모델은 모를 때를 안다: 손으로 고른 임계값 대신 conformal prediction으로 증명 가능한
커버리지 보장 하에 기권한다 — 특히 기존 모델이 전부 조용히 실패하는 불규칙 주기에서."

# mcPHASES 월경주기 4단계 분류 — 프로젝트 전 과정 정리

> Challenge 05 · Foundation Models for Women's Hormonal Health · **Layer 02 (AI Model)**
> 최종 갱신: 2026-07-19 · 정직값(누수 정정·선택편향 제거) 기준

---

## 0. 한눈에 요약 (TL;DR)

- **문제:** mcPHASES에서 월경주기 4단계(월경·난포·가임·황체)를, **호르몬을 보지 않고**(라벨 누수 방지) 예측한다.
- **평가:** Leave-One-Subject-Out(LOSO), 42명, 피험자 부트스트랩 CI로 보고.
- **정직 레퍼런스:** CatBoost + days_since_bleed(flow기반) + HSMM = **macro-F1 0.654 [0.601, 0.701]** → 공개 SOTA(0.662)와 **통계적 동률(on par)**.
- **최고 모델:** + **전용 Fertility detector** = **macro-F1 0.667 [0.620, 0.709] / acc 0.678, Fertility F1 0.531**.
- **핵심 기여 3가지:** ① **전용 2단계 배란 detector**(Fertility 0.462→0.531) ② **Conformal 기권**(증명 가능한 no-call) ③ **오픈 벤치마크 패키지**(고정 LOSO split + model-agnostic CI 하네스).
- **정직성 원칙:** 라벨 누수 발견·정정, 선택편향 제거, 단일점 대신 CI로 보고. 부풀린 0.670은 폐기.

---

## 1. 목표와 위치

챌린지 3레이어 중 **Layer 02 (AI Model)** 선택 — *"reproducibility · scientific rigor · explainability over sheer size."*

채점 3축과 우리 대응:

| 채점축 | 우리 대응 |
|---|---|
| **Women's Health Impact** | 저부담·기기 유연(증상 중심), Fertility(가임) 예측 개선 = 임상적으로 가장 중요한 상 |
| **Technical Excellence** | LOSO·CI 보고, 누수 감사, Conformal 엄밀성, 재현 코드 |
| **Foundation Value** | 고정 split + 평가 하네스를 **오픈 벤치마크**로 공개 → 후속 연구가 즉시 확장 |

**목표가 아닌 것:** N=42에서 +0.01 리더보드 경쟁 / 웨어러블 의존 극대화 / 리키지로 부풀린 점수.

---

## 2. 데이터와 정답 라벨

- **mcPHASES (PhysioNet):** 42명, 2 × 3개월. Fitbit Sense(체온·HR·HRV·수면·호흡) + Dexcom G6 CGM + Mira 호르몬키트(LH/E3G/PdG) + 일일 증상 일지(14종).
- **정답 라벨(phase):** 우리가 만든 게 아니라 **Mira 호르몬 측정값에서 유도된 4단계** ("호르몬 검증 라벨").
- **⚠️ 결정적 함의:** 라벨이 호르몬으로 정의되므로 **lh/estrogen/pdg는 피처에서 제외**(안 그러면 정답 누수).
- **데이터 정합(우리 결정):** 2022 구간만 사용(2024는 증상 99% 결측), 14일 warmup 제거, ≥5일 결측런 행 제외 → master 테이블 2,965행/42명.

---

## 3. 파이프라인 구조 (우리 아키텍처)

```
mcPHASES 원시 CSV (23개 테이블)
      │  build_master.py — 2022 정합 · warmup 제거 · 결측런 제외
      ▼
일 단위 master 테이블
      │  피처 엔지니어링
      │   • days_since_bleed (flow기반, 다중 임계 — 누수 없음)
      │   • rolling mean/std (3·5·7·14일)
      │   • 피험자 baseline 대비 편차(z / q25-delta)
      │   • 배란 앵커(온도·심박 shift)
      ▼
 ┌──────────────────────────┐     ┌───────────────────────────────┐
 │ Stage 1a: CatBoost        │     │ Stage 1b: Fertility detector   │
 │ 4-class per-day 확률       │     │ "가임 vs 나머지" 이진(온도·심박·│
 │                           │     │ 주기위치 20피처)               │
 └────────────┬─────────────┘     └───────────────┬───────────────┘
              │   OOF 확률 0.5 고정 블렌드 (선택편향 없음)         │
              └───────────────┬───────────────────────────────────┘
                              ▼
              Stage 2: HSMM (per-fold Gaussian duration prior)
                   주기 순서(월경→난포→가임→황체) + 지속시간 강제
                              ▼
        ┌─────────────────┬──────────────────┐
        ▼                 ▼                  ▼
   최종 phase 예측    Conformal 기권      SHAP 설명
                   (커버리지 보장 no-call)
```

**설계 원칙:** 모든 단계 OOF(누수 없음), 블렌드 가중치 고정(선택편향 없음), 평가는 LOSO + 피험자 부트스트랩 CI.

---

## 4. 실험 연대기 (정직값 기준)

시간 순. 각 단계가 **무엇을 얼마나** 올렸는지 = 인과의 증거.

| # | 실험 | macro-F1 | 핵심 발견 |
|---|---|---|---|
| 1 | 첫 패스 (B0 증상만) | 0.376 | 파이프라인 end-to-end 작동 확인 |
| 2 | B1 멀티모달 | 0.455 | 웨어러블 +0.08 (증상 위 보조 신호) |
| 3 | 시간 피처(rolling·delta·cycle-day) | 0.525 | **최대 레버리지** — B0도 +0.11 점프 |
| 4 | HSMM 학습형 duration prior | 0.584 | 고정전이 대비 +0.04, 논문 방식 |
| 5 | ⚠️ **누수 발견·정정** | 0.670→**~0.50** | days_since_bleed가 **phase 라벨**에서 계산됨 = 누수. flow기반으로 영구 수정 |
| 6 | 데이터 정합(2022 master) | 0.614 | 증상-빈 2024행 제거만으로 +0.09 |
| 7 | + 배란 앵커 | 0.621 | 소폭(물리 근거) |
| 8 | + HSMM 정합 | 0.644 | 정직 best 갱신 |
| 9 | sin/cos 주기 인코딩 | 0.638 ❌ | **트리엔 무용**(-0.006). dsb 임계분할이 원형을 이미 처리 |
| 10 | + days_since_bleed(flow) | 0.658 | dsb가 최대 레버(importance 1위) |
| 11 | + CatBoost/TabPFN 앙상블 | 0.662* | *가중치를 같은 LOSO에서 선택 → 낙관 편향. **레퍼런스 채택 안 함** |
| 12 | **벤치마크 정직 레퍼런스** | **0.654** | 선택 없는 정직값. CI [0.601, 0.701] = SOTA 동률 |
| 13 | ⭐ **Fertility detector** | **0.667** | Fertility 0.462→**0.531**. CI [0.620, 0.709] |

*11의 0.662는 선택편향이 있어 공식 레퍼런스에서 제외. 정직 레퍼런스는 12의 0.654.

---

## 5. 최종 결과

### 5.1 성능 (LOSO 42명, 피험자 부트스트랩 CI B=2000)

| 모델 | macro-F1 [95% CI] | acc | Fertility F1 |
|---|---|---|---|
| 정직 레퍼런스 (CatBoost+dsb+HSMM) | 0.654 [0.601, 0.701] | 0.671 | 0.498 |
| ⭐ **+ Fertility detector** | **0.667 [0.620, 0.709]** | 0.678 | **0.531** |
| *(참고) 공개 SOTA* | *0.662 [0.618, 0.702]* | *0.676* | *0.462* |

→ **macro-F1은 SOTA와 통계적 동률**(CI 겹침, N=42 노이즈). **차별화는 Fertility F1**: 0.462 → 0.531 (**+0.069**).

### 5.2 클래스별 (Fertility detector)
- 월경 ~0.73 · 난포 ~0.63 · **가임 0.531**(개선) · 황체 ~0.76
- 가임(Fertility)이 여전히 최난이지만, 전용 detector로 유의미하게 회복.

### 5.3 Conformal 기권
- 목표 커버리지 0.90 → 경험적 ~0.898–0.900 (전 백엔드 균일)
- 확신 예측 시 정확도 0.85–0.89, 애매하면 no-call → 불규칙 주기 안전장치.

### 5.4 백엔드 비교 (참고)
CatBoost 0.670* > XGBoost 0.663 > LightGBM 0.662 > **TabPFN 0.644** > EBM 0.617.
→ **주력 = CatBoost 확정.** TabPFN은 N=42에서 트리 못 이김(라이선스·비용 대비 이득 없음) → 각주로만. (*백엔드 비교 수치는 초기 233피처 세팅 기준)

---

## 6. 핵심 발견 & 교훈

1. **누수는 지배 피처에서 온다.** all-data SHAP 1위가 라벨 파생이면 적신호. days_since_bleed를 phase 라벨로 계산 → 0.670으로 부풀려짐. flow 기반으로 고치니 정직값 회복. **교훈: 지배 피처는 항상 ablation.**
2. **가장 큰 레버는 모델이 아니라 데이터·피처.** 데이터 정합 +0.09, 시간피처 +0.11 > 어떤 모델 교체보다 큼.
3. **트리엔 sin/cos 무용.** 트리는 days_since_bleed 임계분할로 주기를 이미 처리. sin/cos는 선형/NN용. *(단, 확인 필요: 일부 문헌은 cycle sin/cos를 상위 피처로 보고 → 구현 차이 점검 여지)*
4. **웨어러블 변동성 법칙이 뒤집힘.** 증상은 변동성(std)이 지배하지만, 웨어러블은 **절대값(황체기 shift)**이 지배(mean 51% > std 38%). 주관 척도 vs 객관 센서의 물리적 차이 = 방어 가능한 발견.
5. **웨어러블은 전체엔 보조, Fertility엔 핵심.** 온도·심박이 배란기 신호를 잡아 전용 detector에서 제값.
6. **N=42에선 단일점이 아니라 CI로.** 0.654 vs 0.662는 노이즈. 정직한 보고 = 신뢰.

---

## 7. 차별점 & 정직성 가드레일

### 우리만의 기여 (재현이 아닌 것)
1. ⭐ **전용 2단계 배란 detector** — 최약 클래스(Fertility)를 분리 이진 모델로 공략 → 0.462→0.531.
2. **Conformal 기권** — 증명 가능한 커버리지 하의 no-call.
3. **오픈 벤치마크 패키지** — 고정 LOSO split + model-agnostic CI 하네스(`benchmark/`).

### 정직하게 지킬 선 (발표/제출 시)
- ❌ *"macro-F1로 SOTA를 이겼다"* — 금지 (CI 겹침).
- ✅ *"Fertility 상(논문 최약점 0.462)을 전용 detector로 0.531까지 개선"* — 이게 헤드라인.
- ✅ 기반 방법(CatBoost+HSMM+cycle-position)은 **확립된 접근의 재현**임을 인정 + 인용 1줄.
- ✅ 성능은 항상 **CI로** 보고.

---

## 8. 산출물 (Deliverables)

| 산출물 | 위치 |
|---|---|
| 데이터 정합 | `ml/mcphases/build_master.py` |
| 메인 파이프라인 | `ml/mcphases/train.py`, `postprocess_hsmm.py`, `conformal.py`, `shap_explain.py` |
| ⭐ Fertility detector | `ml/mcphases/attempt5_fertility_detector.py` |
| 백엔드 비교 | `ai/hormonal/compare.py`, `experiments/hormonal/compare/comparison_report.md` |
| 오픈 벤치마크 | `benchmark/` (README, MIT LICENSE, MODEL_CARD, `splits/loso_folds.json`, `eval/ci_harness.py`, `eval/reference_model.py`) |
| 임상 리포트(설명) | `ai/hormonal/explain.py` (SHAP/conformal → 자연어, template fallback) |
| 실험 로그 | `memory/experiment_log.md` |

---

## 9. 다음 스텝

1. **Fertility detector를 벤치마크 공식 레퍼런스로 승격** (0.667).
2. sin/cos 상충 점검 — 문헌 대비 우리 구현 차이 확인.
3. Fertility 추가 개선 — 배란 전후 급변 피처, class별 임계, per-subject 개인화.
4. 발표/데모 — explain.py 리포트를 데모로, 벤치마크 패키지를 Foundation Value로 포장.

---

## 참고

- 데이터: mcPHASES (PhysioNet, 2025), CC-BY 4.0 · 라벨은 Mira 호르몬 기기 기반.
- 확립된 4단계 SOTA 접근(CatBoost + HSMM, self-report, LOSO 0.662): medRxiv 2026, `10.64898/2026.03.31.26349766` — 기반 방법의 참조점.
- **This is research/education infrastructure. Not a medical device.** N=42 소규모·편향 표본, 불규칙 주기는 별도 보고 및 conformal 기권.

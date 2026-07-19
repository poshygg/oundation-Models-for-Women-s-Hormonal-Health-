# 실험 로그

새 실험을 완료할 때마다 아래 템플릿을 복사해서 상단에 붙임 (최신 순).

---

## 템플릿
### exp/YYMMDD-<name>
- **Reference paper**: source paper found via `python -m harness` (arxiv_id / title)
- **Time budget**: estimate (`time_probe.py` output) vs actual
- **가설**:
- **셋업**: 모델 / 데이터 / 하드웨어 / config 경로
- **결과**: 핵심 수치 3개 이내
- **다음 스텝**:
- **링크**: WandB run URL / commit hash

---

## 실험 기록
<!-- 여기 아래에 최신 실험부터 -->

### exp/260719-LEAKAGE-days_since_bleed ⚠️ 중요 정정
- **발견:** `ai/hormonal/data.py:_days_since_bleed`가 **phase 라벨(phase_idx==Menstrual)**로 월경 시작일을 계산 → **타깃 누수**. 이 피처가 SHAP 1.278로 지배적 1위였음.
- **ablation (LOSO, CatBoost):**
  - full (누수 포함): macroF1 **0.669** / acc 0.672  ← 이전에 "SOTA 도달"이라 보고한 값
  - days_since_bleed 제거: macroF1 **0.496** / acc 0.500  → **-0.173, 누수 기여분**
  - flow_volume 기반(누수 없음) 대체: macroF1 **0.518** / acc 0.525  ← 진짜 정직한 값
- **정정:** 이전 "CatBoost 0.670 = SOTA(0.662) 도달" 주장은 **누수로 부풀려진 것.** 정직한 값은 ~0.50 + flow 기반 cycle 피처가 회복하는 만큼(측정 중). 현재로선 **honest 성능이 SOTA 아래일 가능성 큼.**
- **SOTA 논문은 자가보고 월경(flow)에서 cycle position을 뽑으므로 누수 아님** → 그들 0.662는 유효, 우리가 아직 못 따라간 상태.
- **조치:** (1) data.py를 flow 기반으로 영구 수정, (2) 전체 비교·슬라이드·문서의 0.670 주장 정정, (3) 정직한 baseline에서 다시 레버 쌓기.
- **교훈:** 지배 피처는 항상 누수 의심하고 ablation. all-data SHAP 1위가 라벨 파생이면 적신호.

### exp/260719-backend-compare (lever 2) — ai/hormonal 파이프라인, 실제 데이터, 42-fold LOSO
- **핵심 결과 (macro-F1 / acc, 전부 full LOSO on real mcPHASES, 233 features):**
  - **catboost 0.670 / 0.673** ← best, **공개 SOTA 0.662/0.676 도달·미세 상회**
  - xgboost 0.663 / 0.669
  - lightgbm 0.662 / 0.665
  - **tabpfn 0.644 / 0.652** (GPU, n_est=4) — 트리 못 이김 (N=42 예측 확인)
  - ebm 0.617 / 0.624 (glassbox, 설명 무료)
  - Conformal 커버리지 전 백엔드 ~0.90 균일 (0.898~0.900), decisive acc 0.85~0.89
  - Regression(호르몬 수치): R²~0 (절대수치 복원 불가), Spearman~0.21 (약한 추세)
- **결론:** **주력 = CatBoost 확정.** best 성능 + SOTA 도달 + 빠른 서빙 + SHAP + 라이선스 없음 + HSMM/conformal 이미 통합. TabPFN은 라이선스·CPU/GPU 비용 대비 이득 없음(트리보다 낮음) → "비교 각주"로만. EBM은 glassbox 필요 시 대안.
- **환경:** ai/hormonal이 ml/mcphases보다 피처 엔지니어링 강함(233피처: rolling[3,5]+subject baseline delta(q25)+days_since_bleed, catboost iter600/depth6/l2=6) → 내 초기 ml/mcphases 0.584 대비 0.670. GPU(RTX4060, torch 2.11.0+cu128)로 TabPFN 가속.
- **다음:** CatBoost에 duration-HSMM 얹어 0.670 초과 시도 / Fertility 개선 (lever 3)
- **링크:** `experiments/hormonal/compare/comparison_report.md`, `full_run_gpu.log`

### exp/260719-hsmm-duration (lever 1)

### exp/260719-hsmm-duration (lever 1)
- **가설**: 고정 전이 HSMM → 학습형 Gaussian duration prior(explicit-duration segmental Viterbi)로 교체하면 시간 일관성이 개선된다 (논문 방식)
- **구현**: `ml/mcphases/postprocess_hsmm.py`. 각 phase 실제 run-length에서 Gaussian(mu,sd) 학습(생물학적 population prior). 학습값: 월경 5.6±2.0, 난포 7.1±5.1, 가임 6.5±1.5, 황체 9.6±5.7일.
- **결과** (B1 +temporal 기준): raw 0.525 → **HSMM-duration 0.584 macroF1 / 0.590 acc** (고정전이 best 0.542 대비 +0.04)
  - 클래스별 F1: 월경 0.660, 난포 0.571, **가임 0.436**(최난), 황체 0.607→**0.667**
- **평가**: 지금까지 최고. SOTA 0.662까지 **~0.08**로 좁힘(초기 0.12에서). Fertility가 여전히 병목.
- **다음(남은 레버)**: (2) 분류기 TabPFN/EBM 비교, (3) Fertility 집중(class 임계·배란 전후 급변 피처)
- **링크**: `data/processed/oof_B1_hsmmdur.parquet`

### exp/260719-temporal-features (iteration 2)
- **가설**: 시간 피처 엔지니어링(rolling 3/7일, 전일 delta, 증상 파생 cycle-day)이 격차를 메운다
- **추가 피처**: `add_temporal_features` — TEMPORAL_SYMPTOMS/PHYSIO의 r3/r7/d1 + `days_since_flow`(flow_volume>0에서 파생, 라벨 아님=누수 없음). B0 39피처, B1 191피처.
- **결과** (macro-F1 / accuracy, LOSO):
  - B0 +temporal: **0.486 / 0.485** (이전 0.376 → **+0.11**)
  - B1 +temporal: **0.525 / 0.530** (이전 0.455 → **+0.07**)
  - B1 +temporal +HSMM(stay=0.6): **0.542 / 0.546** ← 현재 best (강한 모델엔 stay 낮춰야 도움; 0.85는 과평활)
  - Conformal(α=0.1) on temporal B1: 커버리지 **0.899**, 확신예측률 0.163(↑), 확신정확도 **0.857**(↑ from 0.82)
- **평가**: 시간 피처가 최대 레버리지 확인. best 0.542 macroF1 (SOTA 0.662까지 ~0.12 남음). HSMM은 이제 시간피처와 부분 중복 → stay 튜닝 필요. Conformal은 모델이 강해질수록 확신 예측 늘고 정확도↑.
- **다음 스텝**: Fertility 클래스가 최난(혼동 최다). 후보: HSMM 학습형 duration prior, 분류기 TabPFN/EBM 비교, class별 임계 조정, 추가 위상 피처.
- **링크**: `ml/mcphases/train.py`(add_temporal_features), `data/processed/oof_B1*.parquet`

### exp/260719-b0b1-pipeline-first-pass

### exp/260719-b0b1-pipeline-first-pass
- **Reference paper**: mcPHASES SOTA (CatBoost+HSMM, LOSO 67.6% acc / 0.662 macro-F1), medRxiv 2026
- **Time budget**: ~데이터조립+B0+HSMM+B1+conformal+SHAP 한 패스 (fold당 CatBoost 0.8s로 최적화; 초기 범주형 target-stats 경로 34s→순서형 정수 인코딩으로 해결)
- **가설**: 멀티모달 웨어러블을 증상 위에 얹으면 symptom-only baseline을 이긴다 / conformal이 정직한 no-call을 준다
- **셋업**: CatBoost(depth5, iter400, Balanced), LOSO 42-fold, 호르몬(lh/estrogen/pdg) 피처 제외(라벨 누수 방지). 코드: `ml/mcphases/{data_assembly,train,postprocess,conformal,shap_explain}.py`
- **결과** (macro-F1 / accuracy, LOSO):
  - B0 증상만: **0.376 / 0.361**
  - B0 + HSMM: **0.420 / 0.404**
  - B1 멀티모달: **0.455 / 0.463**  ← 멀티모달이 B0 대비 +0.08 (가설 검증)
  - B1 + HSMM: **0.481 / 0.490**
  - Conformal(α=0.1): 경험적 커버리지 **정확히 0.900**(phase별 모두 ~0.90), 확신예측 정확도 **0.82**, no-call 0.89
  - SHAP: flow_volume > 안정시심박_z > cramps > 호흡률_z > 심박/HRV_z (황체기 생리 신호 상위 = 생물학 학습 확인)
- **평가**: 파이프라인 end-to-end 작동 확인. 절대 수치는 SOTA(0.662) 미달 — 원인은 **피처 엔지니어링 격차**(논문 83개 엔지니어링 피처 vs 우리 원시+z-score, 증상 결측 41%, HSMM 고정전이). 핵심 결론(멀티모달>증상, HSMM 상승, conformal 커버리지 보장, SHAP 생물학 일치)은 모두 성립.
- **다음 스텝**: (1) 시간 피처 엔지니어링 — 피험자별 rolling 3/7일 평균·delta, cycle-day/days-since-menses 위상 피처, (2) HSMM Gaussian duration prior 학습, (3) B1 분류기 TabPFN/EBM 비교. 격차의 대부분은 (1)에서 메워질 것으로 예상.
- **링크**: OOF 산출물 `data/processed/oof_{B0,B1}*.parquet`, SHAP `data/processed/shap_importance_B1.csv`

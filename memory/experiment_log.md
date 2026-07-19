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

### exp/260719-attempt6-modality-ablation (웨어러블 순수 기여 — 논문 미측정)
- **동일 2022 정합·누수-free·LOSO+HSMM에서 입력 모달리티 비교:**
  - symptoms_only: macroF1 **0.631** (Fert 0.468)
  - wearables_only: **0.466** (Fert 0.336) — 자가보고 없이 약함
  - all(멀티모달): **0.654** (Fert 0.498)
- **웨어러블 순수 기여: macroF1 +0.023, Fertility +0.030.** 논문 간 비교(0.667 vs 0.662)보다 통제된 증거. 논문은 증상-only(0.662)만 보고, 멀티모달 점수 없음(future work).
- **해석:** 웨어러블은 배란기를 가장 크게 도움(+0.030) = 체온/심박 shift. wearables_only 0.466 → **대체 아닌 보완**(flow-의존 논쟁에 정직한 답). caveat: 우리 symptom-only 0.631<논문 0.662(그들 83피처가 더 풍부) → 증상 피처 강화 시 멀티모달 상향 여지.
- **링크:** `ml/mcphases/attempt6_modality_ablation.py`, `experiments/hormonal/attempt6_ablation.log`

### exp/260719-attempt5-fertility-detector ⭐ 최고 정직 모델
- **가설:** 배란기(모든 방법 최약 클래스)를 전용 이진 detector로 공략하면 피처 융합(+0.008)보다 낫다.
- **설계:** main 4-class + "Fertility vs rest" 이진 detector(배란 앵커·온도·심박·주기위치 20피처) 각 LOSO OOF → Fertility 확률 **고정 0.5 블렌드**(선택 편향 없음) → per-fold HSMM. 둘 다 OOF=누수 없음.
- **결과:** detector 단독 Fert-vs-rest F1 0.532. main+HSMM 0.654(Fert .498) → **blended+HSMM macroF1 0.667 / acc 0.678, Fert F1 0.531** (+0.033 Fertility, +0.013 macro).
- **CI (B=2000):** macroF1 0.667 **[0.620, 0.709]** → SOTA 0.662 포함(통계적 동률), 점추정은 미세 상회. 하한 0.620>레퍼런스 0.601.
- **의미:** 논문 명시 최약점(Fert 0.462)을 **전용 2단계 배란 detector**로 공략 = 논문에 없는 아키텍처 차별화. 누수·선택편향 없는 정직값.
- **다음:** 벤치마크 레퍼런스를 이 2단계 모델(0.667)로 교체 검토.
- **링크:** `ml/mcphases/attempt5_fertility_detector.py`, `benchmark/results/oof_fertility_detector.parquet`

### exp/260719-benchmark-package (Foundation Value 딜리버러블)
- **오픈 벤치마크 패키지 `benchmark/`:** README, LICENSE(MIT, mcPHASES raw 제외), MODEL_CARD, `splits/loso_folds.json`(42-fold 고정), `eval/ci_harness.py`(피험자 부트스트랩 CI, model-agnostic), `eval/reference_model.py`(정직 레퍼런스), `results/oof_reference.parquet`.
- **정직 레퍼런스 (누수-free, 선택 편향 없음):** CatBoost + dsb(flow기반) + per-fold HSMM = **macroF1 0.654 / acc 0.671**. per-class: Mens .729 Foll .626 Fert .498 Lute .764.
- **CI (subject bootstrap B=2000):** macroF1 0.654 **[0.601, 0.701]** → **SOTA 0.662 포함 = 통계적 동률(ON PAR).** N=42라 단일점(0.654 vs 0.662)은 노이즈, CI로 보고.
- **주의:** 이전 dsb-ensemble 0.662는 앙상블 가중치·K를 같은 LOSO에서 골라 낙관 편향. 벤치마크는 **선택 없는 정직값 0.654**를 레퍼런스로 채택.
- **링크:** `benchmark/`, `experiments/hormonal/reference_model.log`

### exp/260719-dsb-ensemble (days_since_bleed + 피처축소 + CatBoost/TabPFN 앙상블)
- **Time budget**: est ~15분 vs actual ~11분 (CatBoost full 380s + TabPFN top-25/30 각 ~120s + 앙상블 무료). 게이트 통과.
- **가설**: (1) onset 기반 `days_since_bleed`(flow_volume 파생, 누수 없음)가 any-flow-reset `days_since_flow`보다 강한 cycle 위치 피처다. (2) TabPFN 피처축소+앙상블로 CatBoost+HSMM 0.644를 넘는다.
- **셋업**: master_2022, base+anchor+dsb 146피처, LOSO 42-fold. CatBoost(full) + TabPFN v2(GPU, n_est=8, CatBoost-importance top-25/30). 앙상블=OOF 확률 가중평균(w_cb 스윕) → HSMM. 코드 `ml/mcphases/model_layer_dsb_ensemble.py`.
- **결과 (+HSMM)**:
  - **days_since_bleed가 최대 레버**: CatBoost 0.644 → **0.658** (+0.014), Fertility 0.466 → **0.506**. dsb importance 1위(23.1).
  - `days_since_flow`와 **동일 원천(flow_volume)=중복** (둘 다 넣으면 dsb 17.3/dsf 8.4로 신호 분할). → dsf 제거, dsb만 사용.
  - **best = 앙상블(CB 0.7 / TabPFN-top25 0.3) +HSMM = macroF1 0.662 / acc 0.679** → 공개 honest SOTA(0.662/0.676) **동률·acc 미세 상회.**
  - top-25 > top-30 (0.662 vs 0.656): TabPFN은 피처 더 줄일수록 유리.
- **해석**: win의 대부분은 앙상블이 아니라 **피처(dsb)**. CatBoost+dsb+HSMM 단독 0.658로 95% 확보, 앙상블은 마지막 +0.004. 전부 누수-free 위 정직한 수치.
- **sin/cos**: 다른 창 attempt4에서 트리엔 무용 확인(-0.006) → 본 실험엔 미포함(정합).
- **다음 스텝**: (1) data.py leakage 수정 커밋(전제), (2) AutoTabPFN을 앙상블 파트너로(고비용, parity 이상 불확실), (3) Fertility 전용 detector.
- **링크**: `experiments/hormonal/dsb_only_ensemble.log`

### exp/260719-hsmm-aligned + sincos (attempt 3,4)
- **HSMM 정합(attempt3):** base+anchor 0.621 → **+HSMM 0.644 / acc 0.654** (Mens .762 Foll .609 Fert .468 Lute .736). 지속시간 fold별 학습(누수 없음)+skip_prob. 논문 +0.03과 유사한 +0.023. **정직한 best, SOTA 0.662와 격차 0.018(노이즈).**
- **sin/cos cycle(attempt4) — 실패:** base+anchor+cycle +HSMM **0.638** (-0.006), Fertility 0.468→0.457. **트리엔 sin/cos 무용**(트리는 days_since_flow 임계분할로 원형을 이미 처리). cyc_len=개인지문 노이즈. → **sin/cos는 선형/NN용, 트리엔 폐기.** 다른 창에 "트리면 실측 후 결정" 전달 필요.
- **현재 정직 best = 0.644.** 남은 상승 여지 = Fertility detector(불확실).
- **링크:** `ml/mcphases/attempt3_hsmm.py`, `attempt4_cycle.py`, `experiments/hormonal/attempt{3,4}*.log`

### exp/260719-attempt2-variability (웨어러블 변동성 법칙 검증)
- **가설:** 논문의 "증상 rolling std 지배(45%)" 법칙이 웨어러블에도 성립하나?
- **결과 (웨어러블만, 2022 master):** A 절대값(raw+mean) 0.333 > B 변동성(raw+std) 0.311, C 둘다 0.343. SHAP: **mean 51.4% > std 38.3%** (논문 증상 std 45%와 반대).
- **결론:** **법칙이 뒤집힘.** 웨어러블은 절대값(황체기 shift) > 변동성. 증상=주관적척도→변동성, 웨어러블=객관적센서→절대수준. 물리적으로 타당한 **방어 가능한 차별화 발견**. 실용: mean+std 둘 다 유지(이미 그럼). Attempt 4(SSL)는 데이터 부족으로 스킵 결정.
- **링크:** `ml/mcphases/attempt2_variability.py`, `experiments/hormonal/attempt2.log`

### exp/260719-data-align + attempt1-ovulation-anchor
- **데이터 정합 (2022 master table):** `ml/mcphases/build_master.py` — 2022 한정 + 14일 warmup + 5일 결측런 행제외. 논문 재현뷰 2,965행/42명(논문 2,983/41 근접), 멀티모달뷰 3,110행(웨어러블로 +145 살림).
- **⭐ 최대 발견:** 정합만으로 base macroF1 **0.614** (이전 정직값 ~0.52 → **+0.09**). 원인 = 증상-빈 2024 행 제거. 아직 HSMM 없이도 SOTA 0.662 근접.
- **Attempt 1 (배란 앵커, 인과적 온도/심박 shift 피처):** base 0.614 → +anchor **0.621** (+0.008), Fertility F1 0.453→**0.462** (+0.008). 방향 양성이나 노이즈 경계. Fertility 여전히 병목(논문 0.462와 동일).
- **판단:** ① 2022 마스터 정합을 메인 채택(+0.09), ② 앵커 유지(소폭·물리근거), ③ Fertility엔 **2단계 배란 detector**(피처융합 아닌 분리) 검토.
- **링크:** `ml/mcphases/attempt1_ovulation.py`, `experiments/hormonal/attempt1.log`, `data/processed/mcphases_master_2022.parquet`

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

"""
========================================
제공된 코드 vs 현재 코드 상세 비교
========================================

📌 제공된 코드의 핵심 아이디어
========================================

1. **2단계 ML 파이프라인**
   
   [1단계] Pair Classifier (XGBClassifier)
   - 목적: 어떤 쌍이 공행성 쌍인지 분류
   - 입력: 11개 pair features (max_corr, best_lag, corr_stability 등)
   - 레이블: 상관계수 > 0.38이면 1, 아니면 0
   - 출력: clf_prob (0~1 확률)
   
   [2단계] Value Regressor (XGBRegressor)
   - 목적: 선택된 쌍의 value 예측
   - 입력: 13개 features (현재 코드와 유사)
   - 출력: 다음달 예측값

2. **Union NMAE**
   ```
   U = GT 쌍 ∪ 예측 쌍
   
   각 쌍 (a,b) ∈ U에 대해:
   - (a,b) ∈ GT ∩ 예측 → 실제 예측 오차
   - (a,b) ∈ GT만 또는 예측만 → 오차 = 1.0
   
   NMAE_union = mean(모든 U의 오차)
   ```
   
   → GT를 못 찾은 것도 페널티!

3. **Tau Threshold 최적화**
   - clf_prob >= tau인 쌍만 선택
   - tau를 0.30~0.55 스캔
   - F1 + NMAE 최적화

4. **Negative Sampling**
   - Positive (corr > 0.38): ~수백개
   - Negative (corr < 0.38): ~수천개
   - 2:1 비율로 샘플링 → 밸런스


========================================
현재 코드와의 차이점
========================================

| 항목 | 현재 코드 | 제공 코드 |
|------|-----------|-----------|
| **쌍 선택** | 상관계수 직접 계산 | Pair Classifier 학습 |
| **쌍 개수** | 3,000개 고정 | Tau로 동적 (보통 2,500) |
| **평가 메트릭** | 단순 F1 + NMAE | Union NMAE |
| **특성** | 15개 (decomp 포함) | 13개 (decomp 없음) |
| **검증** | 없음 | Validation 완비 |
| **복잡도** | 낮음 | 높음 (2단계) |


========================================
코드 구조 상세 분석
========================================

✅ **1. build_pair_feature_matrix**
```python
목적: Pair classifier 학습용 데이터 생성

for 각 (leader, follower):
    - lag별 상관계수 계산 (1~7)
    - best_corr, second_corr, corr_stability 등 추출
    
    if abs(best_corr) >= 0.38:
        label = 1  # 공행성 쌍
    else:
        label = 0  # 비공행성 쌍
    
# Negative sampling: pos:neg = 1:2
```

**문제점**: 
- Label 기준(0.38)이 임의적
- GT와 다를 수 있음


✅ **2. train_pair_classifier**
```python
XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.08,
    ...
)

입력: 11개 features
출력: 쌍이 공행성일 확률 (0~1)
```

**장점**: 
- 단순 상관계수보다 정교한 쌍 선택
- 여러 feature 종합 판단

**단점**: 
- 과적합 위험
- 학습 데이터 품질에 의존


✅ **3. score_all_pairs_with_classifier**
```python
전체 품목 쌍(100x99)에 대해:
    - 11개 features 계산
    - clf_prob 예측
    
상위 top_k개 반환 (보통 제한 없음)
```


✅ **4. eval_threshold_union**
```python
tau별로:
    pairs = [clf_prob >= tau인 쌍들]
    
    pred_set = pairs의 집합
    GT_set = validation 실제 쌍
    
    U = GT_set ∪ pred_set
    
    for 각 쌍 in U:
        if 쌍 in GT ∩ pred:
            err = 실제 NMAE
        else:
            err = 1.0  # 페널티
    
    NMAE_union = mean(err)
    Score = 0.6 * F1 + 0.4 * (1 - NMAE_union)
```

**핵심**: Union 개념으로 누락 페널티


========================================
성능 비교 예상
========================================

현재 코드 (Correlation 기반):
✅ 간단하고 직관적
✅ 과적합 위험 낮음
❌ 쌍 선택 정확도 제한적
❌ 검증 없이 제출
→ 예상 점수: 0.35-0.40

제공 코드 (Classifier 기반):
✅ 쌍 선택 정교함
✅ Validation으로 최적화
✅ Union metric 정확
❌ 복잡하고 느림
❌ 과적합 위험
→ 예상 점수: 0.38-0.43 (잘 튜닝시)


========================================
핵심 인사이트
========================================

1. **Pair Classifier의 의미**
   
   현재: 상관계수 > threshold → 쌍
   제공: ML이 쌍 여부 판단
   
   예시:
   - (A,B): corr=0.35, stability=0.8 → clf_prob=0.7 (선택)
   - (C,D): corr=0.40, stability=0.1 → clf_prob=0.3 (제외)
   
   → 단순 threshold보다 똑똑함


2. **Union NMAE의 중요성**
   
   일반 NMAE: 예측한 쌍만 평가
   Union NMAE: GT 못 찾은 것도 페널티
   
   예시:
   GT = {(A,B), (C,D), (E,F)}
   예측 = {(A,B), (G,H)}
   
   U = {(A,B), (C,D), (E,F), (G,H)}
   
   (A,B): 실제 오차 = 0.1
   (C,D): 누락 = 1.0
   (E,F): 누락 = 1.0
   (G,H): 잘못 선택 = 1.0
   
   NMAE_union = (0.1 + 1.0 + 1.0 + 1.0) / 4 = 0.775
   
   → F1 Score와 비슷한 효과!


3. **Tau Optimization**
   
   현재: 고정 개수 (3000개)
   제공: 확률 threshold 최적화
   
   tau=0.3 → 많은 쌍, 낮은 precision
   tau=0.5 → 적은 쌍, 높은 precision
   
   → Validation으로 최적 tau 찾기


========================================
적용 추천 여부
========================================

🤔 **적용해야 할까?**

YES, 다음 경우:
✅ Validation 데이터 있음
✅ 시간 여유 충분 (2-3시간)
✅ 현재 점수 < 0.35
✅ 복잡한 모델 OK

NO, 다음 경우:
❌ 빨리 제출해야 함
❌ 현재 점수 >= 0.38
❌ 단순한 방법 선호
❌ 디버깅 시간 없음


🎯 **절충안: 하이브리드**

1. Pair 선택: 제공 코드의 Classifier
2. Value 예측: 현재 코드 (15 features)
3. Validation: Union NMAE
4. 최종: Tau 최적화

→ 두 방법의 장점 결합


========================================
실행 가이드
========================================

제공 코드 실행하려면:

```python
# Validation (로컬 검증)
run_validation()
# → tau별 F1, NMAE, Score 출력
# → 최적 tau 확인

# Submission (제출 파일 생성)
run_submission("submission_classifier.csv")
# → 2500개 쌍, 예측값
```

현재 점수(0.319)보다는 개선 가능성 높음


========================================
최종 판단
========================================

**제공 코드는 고급 기법**
- Kaggle 상위권 전략
- 2단계 ML 파이프라인
- Union metric으로 엄격한 평가

**하지만 복잡함**
- 디버깅 어려움
- 과적합 위험
- 하이퍼파라미터 많음

**추천**:
1. 먼저 현재 코드로 0.35+ 달성 시도
2. 안 되면 제공 코드 실행
3. 둘 다 제출해서 비교

**예상**:
- 현재 코드: 0.35-0.38
- 제공 코드: 0.38-0.42
- 앙상블: 0.40-0.44
"""

print(__doc__)

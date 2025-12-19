import pandas as pd
import numpy as np

print("="*80)
print("0.1272 점수 긴급 진단")
print("="*80)

# 제출 파일 로드
df = pd.read_csv('./submission_2stage_classifier.csv')

print("\n【1. 기본 검증】")
print(f"쌍 개수: {len(df):,}")
print(f"컬럼: {df.columns.tolist()}")
print(f"중복 쌍: {df.duplicated(['leading_item_id', 'following_item_id']).sum()}개")

print("\n【2. 값 분포 문제】")
print(df['value'].describe())
print(f"\n0인 값: {(df['value'] == 0).sum()}개 ({(df['value']==0).sum()/len(df)*100:.1f}%)")
print(f"음수: {(df['value'] < 0).sum()}개")
print(f"평균: {df['value'].mean():,.0f}")
print(f"중앙값: {df['value'].median():,.0f}")

# 실제 데이터와 비교
train = pd.read_csv('./train.csv')
print(f"\n실제 데이터 평균: {train['value'].mean():,.0f}")
print(f"제출 / 실제 비율: {df['value'].mean() / train['value'].mean():.2f}")

print("\n【3. 쌍 타입 분석】")
print(f"Leading 품목 수: {df['leading_item_id'].nunique()}")
print(f"Following 품목 수: {df['following_item_id'].nunique()}")

print("\n【4. Lag 분포 확인】")
# classifier_approach.py의 쌍 선택 로직 확인
print("Classifier에서 선택한 쌍의 특성:")
print(f"최소 확률 기준: 0.0004 (너무 낮음!)")
print(f"→ 거의 무작위 쌍 선택됨")

print("\n" + "="*80)
print("【추정 문제점】")
print("="*80)
print("\n1. ❌ Classifier 확률 분포 문제")
print("   - 대부분 확률이 0.0001~0.0005 (매우 낮음)")
print("   - Tau threshold 스캔에서 0.30~0.50 모두 1,670개")
print("   - → Classifier가 제대로 학습 안 됨")

print("\n2. ❌ Positive/Negative 불균형")
print("   - Positive: 1,668개 (16.85%)")
print("   - Negative: 5,004개 (3배 샘플링)")
print("   - → 클래스 불균형으로 학습 실패")

print("\n3. ❌ Correlation threshold 0.35 너무 높음")
print("   - Positive 라벨이 너무 적음")
print("   - 실제 공행성 쌍을 놓침")

print("\n4. ❌ Value Regressor 문제")
print("   - 최근 월 평균값으로 학습")
print("   - 실제 공행성 값이 아님")
print("   - → 잘못된 타겟으로 학습")

print("\n" + "="*80)
print("【해결 방안】")
print("="*80)
print("\n✅ 방안 1: submission_v3_final.csv 제출")
print("   - Correlation 기반 (Lag 0 우선)")
print("   - 예상: 0.35-0.37")
print("   - 가장 안정적")

print("\n✅ 방안 2: Classifier 수정 후 재실행")
print("   - Threshold 0.35 → 0.20")
print("   - Value 타겟 수정 (실제 공행성 값)")
print("   - Negative sampling 비율 조정")

print("\n✅ 방안 3: 단순 상관계수 기반")
print("   - Lag 0 상관계수만 사용")
print("   - 상위 3,000개")
print("   - 예상: 0.32-0.35")

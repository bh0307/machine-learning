import pandas as pd

files = {
    'v3_final': 'submission_v3_final.csv',
    '2stage_classifier': 'submission_2stage_classifier.csv',
    'ensemble': 'submission_ensemble.csv'
}

print("="*80)
print("최종 제출 파일 비교")
print("="*80)

for name, fname in files.items():
    df = pd.read_csv(f'./{fname}')
    print(f"\n【{name}】 - {fname}")
    print(f"  쌍 개수: {len(df):,}")
    print(f"  평균값: {df['value'].mean():,.0f}")
    print(f"  중앙값: {df['value'].median():,.0f}")
    print(f"  최소값: {df['value'].min():,}")
    print(f"  최대값: {df['value'].max():,}")
    print(f"  0개: {(df['value']==0).sum()}개")

print("\n" + "="*80)
print("제출 추천 순서")
print("="*80)
print("\n1순위: submission_2stage_classifier.csv")
print("   - 0.3495 코드 기반 2-Stage ML")
print("   - Pair Classifier + Value Regressor")
print("   - 예상: 0.36-0.42")
print("\n2순위: submission_v3_final.csv")
print("   - Lag 0-3 가중 상관계수")
print("   - Lag 0 우선 (65.3%)")
print("   - 예상: 0.35-0.37")
print("\n3순위: submission_ensemble.csv")
print("   - v3_final + improved_v2 앙상블")
print("   - 예상: 0.36-0.38")

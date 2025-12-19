import pandas as pd
import os

print("="*80)
print("앙상블 제출 파일 생성")
print("="*80)

# 두 제출 파일 로드
files_to_ensemble = []
file_names = []

# v3_final (Correlation 기반)
if os.path.exists('./submission_v3_final.csv'):
    df_v3 = pd.read_csv('./submission_v3_final.csv')
    files_to_ensemble.append(df_v3)
    file_names.append('Correlation (v3_final)')
    print(f"✓ Correlation 기반: {len(df_v3):,}개 쌍, 평균 {df_v3['value'].mean():,.0f}")

# 0.3495 점수 파일 찾기 (improved_v2 사용 - 안정적)
classifier_candidates = [
    'submission_improved_v2.csv',  # 평균 3.7M, 최대 118M (합리적)
    'submission_advanced.csv',     # 평균 3.9M
    'submission_improved.csv'
]

for fname in classifier_candidates:
    if os.path.exists(f'./{fname}'):
        df_classifier = pd.read_csv(f'./{fname}')
        files_to_ensemble.append(df_classifier)
        file_names.append(f'Classifier ({fname})')
        print(f"✓ Classifier 기반: {len(df_classifier):,}개 쌍, 평균 {df_classifier['value'].mean():,.0f}")
        break

if len(files_to_ensemble) < 2:
    print("⚠️ Classifier 제출 파일을 찾을 수 없습니다")
    print("   → Correlation 단독으로 사용하세요: submission_v3_final.csv")
    exit()

print(f"\n총 {len(files_to_ensemble)}개 파일 앙상블")

# 쌍 합집합 구하기
print("\n【1단계: 쌍 합집합 구성】")
all_pairs = set()
for df in files_to_ensemble:
    for _, row in df.iterrows():
        pair = (row['leading_item_id'], row['following_item_id'])
        all_pairs.add(pair)

print(f"  전체 유니크 쌍: {len(all_pairs):,}개")

# 앙상블 전략: merge로 빠르게
print("\n【2단계: 값 앙상블 (평균)】")

# 각 파일에 인덱스 추가
for i, df in enumerate(files_to_ensemble):
    df['pair_key'] = df['leading_item_id'] + '_' + df['following_item_id']

# 첫 파일을 기준으로 시작
df_ensemble = files_to_ensemble[0][['leading_item_id', 'following_item_id', 'value', 'pair_key']].copy()
df_ensemble.rename(columns={'value': 'value_0'}, inplace=True)

# 나머지 파일들 merge
for i in range(1, len(files_to_ensemble)):
    df_temp = files_to_ensemble[i][['pair_key', 'value']].copy()
    df_temp.rename(columns={'value': f'value_{i}'}, inplace=True)
    df_ensemble = df_ensemble.merge(df_temp, on='pair_key', how='outer')

# 평균 계산
value_cols = [col for col in df_ensemble.columns if col.startswith('value_')]
df_ensemble['value'] = df_ensemble[value_cols].mean(axis=1).astype(int)

# 필요한 컬럼만 선택
df_ensemble = df_ensemble[['leading_item_id', 'following_item_id', 'value']].dropna()

# 3,000개 제한
if len(df_ensemble) > 3000:
    print(f"\n【3단계: 상위 3,000개 선택】")
    # value 기준 정렬 후 상위 선택
    df_ensemble = df_ensemble.nlargest(3000, 'value')
    print(f"  {len(all_pairs):,}개 → 3,000개 선택")

print(f"\n【최종 앙상블 통계】")
print(df_ensemble['value'].describe())
print(f"\n평균: {df_ensemble['value'].mean():,.0f}")
print(f"쌍 개수: {len(df_ensemble):,}")

# 저장
df_ensemble.to_csv('./submission_ensemble.csv', index=False)

print("\n" + "="*80)
print("✅ 앙상블 제출 파일 생성 완료!")
print("="*80)
print(f"\n📁 파일명: submission_ensemble.csv")
print(f"📊 쌍 개수: {len(df_ensemble):,}개")
print(f"💰 평균 예측값: {df_ensemble['value'].mean():,.0f}")
print(f"\n🎯 사용 방법:")
for i, name in enumerate(file_names, 1):
    print(f"   {i}. {name}")
print(f"\n📈 예상 점수: 0.37-0.40")
print(f"   (개별 모델보다 안정적)")

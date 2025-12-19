import pandas as pd

df = pd.read_csv('granger_results.csv')

print('='*80)
print('Granger Causality 전체 분석 결과')
print('='*80)

print(f'\n총 쌍 개수: {len(df):,}개')

print('\n【P-value 통계】')
print(df['p_value'].describe())

print('\n【P-value 범위별 분포】')
print(f'  P < 0.001 (매우매우 강함): {(df.p_value < 0.001).sum():,}개 ({(df.p_value < 0.001).sum()/len(df)*100:.1f}%)')
print(f'  P < 0.01 (매우 강함): {(df.p_value < 0.01).sum():,}개 ({(df.p_value < 0.01).sum()/len(df)*100:.1f}%)')
print(f'  P < 0.05 (강함): {(df.p_value < 0.05).sum():,}개 ({(df.p_value < 0.05).sum()/len(df)*100:.1f}%)')
print(f'  P < 0.10 (유의미): {(df.p_value < 0.10).sum():,}개 ({(df.p_value < 0.10).sum()/len(df)*100:.1f}%)')

print('\n【Lag 분포】')
lag_dist = df['lag'].value_counts().sort_index()
for lag, cnt in lag_dist.items():
    print(f'  Lag {lag}: {cnt:,}개 ({cnt/len(df)*100:.1f}%)')

print('\n【Causality Score 통계】')
print(df['causality_score'].describe())

print('\n【품목별 통계】')
print(f'  선행 품목 수: {df.leading_item_id.nunique()}개')
print(f'  후행 품목 수: {df.following_item_id.nunique()}개')

leader_counts = df.leading_item_id.value_counts()
print(f'  평균 선행관계 수/품목: {leader_counts.mean():.1f}개')
print(f'  최다 선행 품목: {leader_counts.index[0]} ({leader_counts.iloc[0]}개 후행품목에 영향)')

follower_counts = df.following_item_id.value_counts()
print(f'  평균 후행관계 수/품목: {follower_counts.mean():.1f}개')
print(f'  최다 후행 품목: {follower_counts.index[0]} ({follower_counts.iloc[0]}개 선행품목에 영향받음)')

print('\n【최강 인과관계 TOP 20】 (낮은 p-value)')
print(df.nsmallest(20, 'p_value')[['leading_item_id', 'following_item_id', 'lag', 'p_value', 'causality_score']].to_string(index=False))

print('\n【약한 인과관계 BOTTOM 10】 (높은 p-value)')
print(df.nlargest(10, 'p_value')[['leading_item_id', 'following_item_id', 'lag', 'p_value', 'causality_score']].to_string(index=False))

print('\n【Lag별 평균 p-value】')
lag_pvalue = df.groupby('lag')['p_value'].agg(['mean', 'median', 'min', 'max', 'count'])
print(lag_pvalue.to_string())
